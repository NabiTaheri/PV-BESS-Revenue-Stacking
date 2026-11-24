"""Exact, sparse, reusable MILP evaluator for repeated S2 opportunity-cost calculations.

This is NOT an approximation. It solves the *same* 96-slot SciPy/HiGHS MILP as
DA/da_mfrr_energy.py, with its exact objective, integrality, bounds and
constraints. Static constraints and forecast preprocessing are built once per
capacity-stage day. For each changed reservation only the affected variable
bounds / six per-interval inequality bounds / two binary coefficients change.

No original file is modified; no input data, scenario or seed is resampled.
Use 'python validate_exact_speed.py' before publishing or large runs.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


class ExactSparseS2:
    def __init__(self, capacity_module):
        self.module = capacity_module
        market_config, engine, _ = capacity_module._load_s2_engine()
        self.engine = engine
        df_fc, _ = capacity_module.load_forecast_input_data(int(capacity_module.build_cfg()['HORIZON']))
        self.timestamps = pd.to_datetime(df_fc['timestamp']).reset_index(drop=True)
        n = len(df_fc)
        battery = capacity_module.build_cfg()['BATTERY']
        baseline = capacity_module.make_free_battery_reservation_profile(self.timestamps, capacity_module.build_cfg())

        # Use the original parsing code ONCE to reproduce its precise fallbacks,
        # clipping rules, datetime handling, and MarketConfig overrides.
        with tempfile.TemporaryDirectory(prefix='exact_sparse_init_') as temp:
            t = Path(temp)
            fc_file = t/'forecasts_rolling.csv'
            actuals_file = t/'actuals_forecast_as_actuals.csv'
            reservation_file = t/'reservation.csv'
            df_fc.to_csv(fc_file,index=False,encoding='utf-8-sig')
            capacity_module._forecast_as_actuals_file(df_fc,actuals_file)
            baseline.to_csv(reservation_file,index=False,encoding='utf-8-sig')
            raw_config = market_config.MarketConfig()
            supported = {f.name for f in fields(raw_config)} if is_dataclass(raw_config) else set()
            kwargs = {
                'forecast_file':fc_file, 'actuals_file':actuals_file,
                'reservation_file':reservation_file,
                'output_report':t/'tmp_report.csv', 'output_summary':t/'tmp_summary.csv',
                'soc_init_pct':float(getattr(capacity_module.cfg_module,'SOC_INIT_PCT',50.0)),
                'soc_target_pct':float(getattr(capacity_module.cfg_module,'SOC_TARGET_PCT',50.0)),
                'rolling_soc_reference_pct':float(getattr(capacity_module.cfg_module,'SOC_TARGET_PCT',50.0)),
                'rolling_recovery_target_soc_pct':float(getattr(capacity_module.cfg_module,'SOC_TARGET_PCT',50.0)),
                'soc_terminal_sync':bool(getattr(capacity_module.cfg_module,'SOC_TERMINAL_SYNC',True)),
                'rolling_terminal_soc_sync':bool(getattr(capacity_module.cfg_module,'SOC_TERMINAL_SYNC',True)),
                'enable_mfrr_energy_market':True,
                'include_hypothetical_mfrr_in_da_stage':True,
                'reservation_file_required':True,
                'allow_reservation_row_order_fallback':False,
                'print_interval_table':False,
            }
            config = replace(raw_config, **{key:value for key,value in kwargs.items() if key in supported})
            engine.apply_config(config)
            df = engine.load_data()
        if len(df)!=n:
            raise ValueError('Forecast input merged to an unexpected number of rows')
        self.n=n
        self.df=df
        self.vi=engine._VarIndex(n)
        self.mfrr_alpha=float(np.clip(engine.MFRR_ACTIVATION_FRACTION,0,1))
        self.balancing_enabled=bool(engine.ENABLE_BALANCING_MARKET)
        self.grid_limit=float(engine.GRID_LIMIT_MW)
        self.p_max=float(engine.BATTERY_P_MAX_MW)
        self.energy_mwh=float(engine.BATTERY_E_MWH)
        self.eff=float(engine.EFF)
        self.inv_eff=float(engine.INV_EFF)
        self.dt=float(engine.DT_H)
        self.deg_cost=float(engine.CYCLE_COST_EUR_PER_MWH)/2
        self.grid_charging=bool(engine.GRID_CHARGING_ALLOWED)
        self.pv=df['pv_forecast_mw'].to_numpy(float)
        self.da_price=df['da_price_forecast'].to_numpy(float)
        self.bal_price=df['balancing_price_forecast'].to_numpy(float)
        self.flag=df['balancing_flag_forecast'].to_numpy(float)
        self.init_soc=float(engine.SOC_INIT_PCT)/100*self.energy_mwh
        self.target_soc=float(engine.SOC_TARGET_PCT)/100*self.energy_mwh
        self.terminal_sync=bool(engine.SOC_TERMINAL_SYNC)
        self._compile()
        self.calls=0
        self.total_solver_seconds=0.
        self.reused_optimum=0
        self._reference_solution=None
        self._last_solution=None

    def _compile(self):
        n=self.n; vi=self.vi; k=self.mfrr_alpha
        nc=vi.total
        c=np.zeros(nc,float)
        integrality=np.zeros(nc,np.int8)
        lower=np.zeros(nc,float)
        upper=np.full(nc,np.inf,float)
        rr=[];cc=[];vv=[]
        eq_rhs=np.zeros(2*n,float)
        # Row order: all equalities, then six inequalities per interval.
        # Exactly the original model; sparse storage merely eliminates zeros.
        def put(r,v,value):
            rr.append(int(r));cc.append(int(v));vv.append(float(value))
        for t in range(n):
            i=lambda name:vi.x(name,t)
            pv=self.pv[t]; da=self.da_price[t];bp=self.bal_price[t]
            c[i('pv_export')]=-da*self.dt
            c[i('battery_discharge')]=-da*self.dt+self.deg_cost*self.dt
            c[i('grid_charge')]=da*self.dt+self.deg_cost*self.dt
            c[i('pv_charge')]=self.deg_cost*self.dt
            c[i('mfrr_up')]=-k*bp*self.dt+k*self.deg_cost*self.dt
            c[i('mfrr_down')]= k*bp*self.dt+k*self.deg_cost*self.dt
            integrality[i('charge_mode')]=1
            upper[i('pv_export')]=pv
            upper[i('pv_curtail')]=pv
            upper[i('charge_mode')]=1.0
            if not self.balancing_enabled or k<=float(self.engine.EPS):
                upper[i('mfrr_up')]=upper[i('mfrr_down')]=0.0
            # Equalities.
            put(2*t,i('pv_export'),1);put(2*t,i('pv_charge'),1);put(2*t,i('pv_curtail'),1)
            eq_rhs[2*t]=pv
            r=2*t+1
            for j,v in [(vi.soc(t+1),1),(vi.soc(t),-1),
                        (i('grid_charge'),-self.eff*self.dt),
                        (i('pv_charge'),-self.eff*self.dt),
                        (i('mfrr_down'),-self.eff*self.dt*k),
                        (i('battery_discharge'),self.inv_eff*self.dt),
                        (i('mfrr_up'),self.inv_eff*self.dt*k)]:put(r,j,v)
            # Inequalities.
            r=2*n+6*t
            for j,v in [(i('grid_charge'),1),(i('pv_charge'),1),(i('mfrr_down'),k)]:put(r,j,v)
            for j,v in [(i('battery_discharge'),1),(i('mfrr_up'),k)]:put(r+1,j,v)
            for j,v in [(i('pv_export'),1),(i('battery_discharge'),1),(i('mfrr_up'),k)]:put(r+2,j,v)
            for j,v in [(i('grid_charge'),1),(i('mfrr_down'),k)]:put(r+3,j,v)
            for j,v in [(i('grid_charge'),1),(i('pv_charge'),1),(i('mfrr_down'),1)]:put(r+4,j,v)
            put(r+4,i('charge_mode'),-1.0) # coefficient updated with -p_charge
            for j,v in [(i('battery_discharge'),1),(i('mfrr_up'),1)]:put(r+5,j,v)
            put(r+5,i('charge_mode'),1.0) # updated with p_discharge
        a=coo_matrix((vv,(rr,cc)),shape=(8*n,nc),dtype=float).tocsr()
        a.sort_indices()
        mode_charge_positions=np.empty(n,dtype=np.int64)
        mode_dis_positions=np.empty(n,dtype=np.int64)
        for t in range(n):
            col=vi.x('charge_mode',t)
            for row,dest in ((2*n+6*t+4,mode_charge_positions),
                             (2*n+6*t+5,mode_dis_positions)):
                start,end=a.indptr[row],a.indptr[row+1]
                idx=np.searchsorted(a.indices[start:end],col)
                if idx>=end-start or a.indices[start+idx]!=col:
                    raise RuntimeError('Could not locate dynamic mode coefficient')
                dest[t]=start+idx
        self.A=a
        self.mode_charge_positions=mode_charge_positions
        self.mode_dis_positions=mode_dis_positions
        self.c=c
        self.integrality=integrality
        self.base_lb=lower
        self.base_ub=upper
        self.base_rows_lower=np.concatenate([eq_rhs,np.full(6*n,-np.inf)])
        self.base_rows_upper=np.concatenate([eq_rhs,np.full(6*n,np.inf)])

    def _extract_reservation(self,res):
        if len(res)!=self.n:
            raise ValueError('Reservation length does not match daily horizon')
        if 'timestamp' not in res:
            raise ValueError('Timestamp required for exact timestamp match')
        if not np.array_equal(pd.to_datetime(res['timestamp']).to_numpy(),
                              self.timestamps.to_numpy()):
            raise ValueError('Reservation timestamps do not match forecast timestamps')
        df=self.df
        def read(name,default):
            if name not in res:return np.full(self.n,float(default))
            return pd.to_numeric(res[name],errors='coerce').fillna(default).to_numpy(float)
        p_ch=np.clip(read('p_charge_max_mw',self.p_max),0,self.p_max)
        p_dis=np.clip(read('p_discharge_max_mw',self.p_max),0,self.p_max)
        soc_min=np.clip(read('soc_min_pct',self.engine.PHYSICAL_SOC_MIN_PCT),0,100)/100*self.energy_mwh
        soc_max=np.clip(read('soc_max_pct',self.engine.PHYSICAL_SOC_MAX_PCT),0,100)/100*self.energy_mwh
        return p_ch,p_dis,soc_min,soc_max

    def value(self,reservation):
        p_ch,p_dis,soc_min,soc_max=self._extract_reservation(reservation)
        n=self.n;vi=self.vi;k=self.mfrr_alpha
        ub=self.base_ub.copy()
        lb=self.base_lb.copy()
        ru=self.base_rows_upper.copy()
        a=self.A.copy()
        a.data[self.mode_charge_positions]=-p_ch
        a.data[self.mode_dis_positions]=p_dis
        for t in range(n):
            ub[vi.x('grid_charge',t)]=min(p_ch[t],self.grid_limit) if self.grid_charging else 0.
            ub[vi.x('pv_charge',t)]=min(p_ch[t],self.pv[t])
            ub[vi.x('battery_discharge',t)]=p_dis[t]
            ub[vi.x('mfrr_up',t)]=p_dis[t] if self.balancing_enabled and k>float(self.engine.EPS) and self.flag[t]>0 else 0.
            ub[vi.x('mfrr_down',t)]=p_ch[t] if self.balancing_enabled and k>float(self.engine.EPS) and self.flag[t]<0 else 0.
            r=2*n+6*t
            ru[r:r+6]=[p_ch[t],p_dis[t],self.grid_limit,self.grid_limit,0.,p_dis[t]]
        lb[vi.soc(0)]=ub[vi.soc(0)]=self.init_soc
        for t in range(1,n+1):
            if t==n and self.terminal_sync:
                lb[vi.soc(t)]=ub[vi.soc(t)]=self.target_soc
            else:
                idx=min(t,n-1) # Original uses t's interval (not t-1).
                lb[vi.soc(t)]=soc_min[idx]
                ub[vi.soc(t)]=soc_max[idx]
        # Exact, mathematically certified short circuit: if an optimal schedule
        # for a RELAXATION of this reservation is still feasible under the
        # requested tighter bounds, it remains optimal. The objective is identical to the reference solve.
        # The first unconstrained S2 optimum can cover many FCR intervals.
        for candidate in (self._reference_solution,self._last_solution):
            if candidate is None:continue
            prev_pch,prev_pdis,prev_smin,prev_smax,prev_x,prev_value = candidate
            if not (np.all(prev_pch+1e-10>=p_ch) and
                    np.all(prev_pdis+1e-10>=p_dis) and
                    np.all(prev_smin-1e-10<=soc_min) and
                    np.all(prev_smax-1e-10>=soc_max)):
                continue
            # Re-check all original dynamic constraints, not only power/SOC.
            if np.any(prev_x<lb-1e-7) or np.any(prev_x>ub+1e-7):
                continue
            ax=a@prev_x
            if np.any(ax<self.base_rows_lower-1e-7) or np.any(ax>ru+1e-7):
                continue
            self.calls+=1
            self.reused_optimum+=1
            # Since new feasible set is a subset of candidate's set and old
            # maximizer is feasible in it, its objective is EXACTLY unchanged.
            return prev_value
        t0=time.perf_counter()
        result=milp(c=self.c,integrality=self.integrality,
                    bounds=Bounds(lb,ub),
                    constraints=LinearConstraint(a,self.base_rows_lower,ru),
                    options={'time_limit':300.0,'mip_rel_gap':1e-9})
        self.calls+=1
        self.total_solver_seconds+=time.perf_counter()-t0
        if not result.success:
            raise RuntimeError('Exact sparse S2 MILP failed: '+str(result.message))
        val=-float(result.fun)
        candidate=(p_ch.copy(),p_dis.copy(),soc_min.copy(),soc_max.copy(),result.x.copy(),val)
        if self._reference_solution is None:
            self._reference_solution=candidate
        else:
            self._last_solution=candidate
        return val
