"""Full-distance bounded exhaustive planner, inspired by the separate MAS search.

Original implementation: enumerate feasible tyre/stint states, use cumulative
lap costs and paired uncertainty. No scripted rivals or 20-lap compression.
"""
from dataclasses import dataclass, field, asdict
from itertools import product, combinations
from collections import Counter
import numpy as np
from .features import COMPOUNDS


@dataclass(frozen=True)
class EventPolicy:
    laps: int
    pit_loss: float = 22.
    pit_sigma: float = 3.
    min_stint: int = 5
    max_stops: int = 2
    inventory: dict = field(default_factory=lambda: {"SOFT":1, "MEDIUM":2, "HARD":2})
    pit_source: str = "Owner scenario assumption; not measured for this event"
    inventory_source: str = "Assumed available fresh sets: 1 soft, 2 medium, 2 hard; not verified allocation"

    def __post_init__(self):
        if not 6 <= self.laps <= 90 or not 1 <= self.min_stint <= self.laps//2:
            raise ValueError("Invalid race distance or minimum stint")
        if self.max_stops not in (1, 2) or not np.isfinite(self.pit_loss) or not 5 <= self.pit_loss <= 60:
            # A wider 100s pit-loss is useful for explicit sensitivity tests/scenarios.
            if self.max_stops not in (1,2) or not np.isfinite(self.pit_loss) or not 5 <= self.pit_loss <= 120:
                raise ValueError("Invalid pit policy")
        if not np.isfinite(self.pit_sigma) or not 0 <= self.pit_sigma <= 20:
            raise ValueError("Invalid pit uncertainty")
        if set(self.inventory)-set(COMPOUNDS) or any(not isinstance(n,int) or not 0<=n<=3 for n in self.inventory.values()):
            raise ValueError("Invalid inventory")


def validate_plan(plan, policy):
    comps, pits, lengths = plan["compounds"], plan["pits"], plan["lengths"]
    boundaries = [0]+pits+[policy.laps]
    if (len(comps) != len(pits)+1 or len(lengths)!=len(comps)
            or not 1<=len(pits)<=policy.max_stops or len(set(comps))<2
            or any(not isinstance(p,int) for p in pits)
            or lengths != [b-a for a,b in zip(boundaries,boundaries[1:])]
            or any(n<policy.min_stint for n in lengths)
            or any(n>policy.inventory.get(c,0) for c,n in Counter(comps).items())):
        raise ValueError("Illegal tyre plan")


def enumerate_plans(policy):
    output=[]
    for stops in range(1,policy.max_stops+1):
        splits = []
        for pits in combinations(range(policy.min_stint, policy.laps-policy.min_stint+1),stops):
            bounds=(0,)+pits+(policy.laps,)
            lengths=[b-a for a,b in zip(bounds,bounds[1:])]
            if min(lengths)>=policy.min_stint:
                splits.append((list(pits),lengths))
        for compounds in product(COMPOUNDS,repeat=stops+1):
            if len(set(compounds))<2 or any(n>policy.inventory.get(c,0) for c,n in Counter(compounds).items()):
                continue
            output.extend(dict(compounds=list(compounds),pits=pits,lengths=lengths) for pits,lengths in splits)
    return output


def rank_strategies(model, driver, policy, seed=42, degradation=1., pit_adjustment=0.):
    d=model["drivers"][driver]
    if d["base_seconds"] is None:
        return []
    plans=enumerate_plans(policy)
    if not plans:
        return []
    n=len(plans)
    # Total number of laps and cumulative tyre ages per compound per candidate.
    laps=np.zeros((n,3)); ages=np.zeros((n,3))
    stops=np.empty(n)
    for j,p in enumerate(plans):
        stops[j]=len(p["pits"])
        for c,length in zip(p["compounds"],p["lengths"]):
            i=COMPOUNDS.index(c)
            laps[j,i]+=length; ages[j,i]+=length*(length+1)/2
    offsets=np.array([d["compounds"][c]["offset"] for c in COMPOUNDS])
    slopes=np.array([d["compounds"][c]["slope"] for c in COMPOUNDS])*degradation
    constant=policy.laps*d["base_seconds"]+d["progression"]*policy.laps*(policy.laps+1)/2
    costs=constant+laps@offsets+ages@slopes+stops*(policy.pit_loss+pit_adjustment)
    best_by_sequence={}
    for index in np.argsort(costs,kind="stable"):
        sequence=tuple(plans[index]["compounds"])
        if sequence not in best_by_sequence:
            best_by_sequence[sequence]=int(index)
            if len(best_by_sequence)==3:
                break
    selected=list(best_by_sequence.values())
    rng=np.random.default_rng(seed)
    # These draws are SHARED by all candidates. Pace offsets persist for the race.
    draws=256
    pace_draw=rng.normal(0,d["pace_sigma"],draws)*policy.laps
    lap_draw=rng.normal(0,d["lap_sigma"]*np.sqrt(policy.laps),draws)
    slope_draw=rng.normal(size=(3,draws))*np.array([d["compounds"][c]["slope_sigma"] for c in COMPOUNDS])[:,None]*degradation
    # Positive pace degradation is a modelling constraint, consistently enforced in draws.
    # Lognormal slopes preserve the same positive expected slopes used in ranking.
    relative=np.array([d["compounds"][c]["slope_sigma"] for c in COMPOUNDS])*degradation/np.maximum(slopes,.001)
    log_sigma=np.sqrt(np.log1p(relative**2))
    slope_draw=slopes[:,None]*(np.exp(rng.normal(size=(3,draws))*log_sigma[:,None]-.5*log_sigma[:,None]**2)-1)
    pit_draw=rng.normal(0,policy.pit_sigma,draws)
    values={i:costs[i]+pace_draw+lap_draw+ages[i]@slope_draw+stops[i]*pit_draw for i in selected}
    output=[]
    for i in selected:
        p=plans[i]
        eligible=[j for j,v in enumerate(plans) if v["compounds"]==p["compounds"] and costs[j]<=costs[i]+1.]
        windows=[[min(plans[j]["pits"][k] for j in eligible),max(plans[j]["pits"][k] for j in eligible)] for k in range(len(p["pits"]))]
        delta=values[i]-values[selected[0]]
        quant=np.quantile(values[i],[.1,.5,.9]); dq=np.quantile(delta,[.1,.9])
        output.append(p|dict(mean=round(float(costs[i]),2),median=round(float(quant[1]),2),
                             p10=round(float(quant[0]),2),p90=round(float(quant[2]),2),
                             delta=round(float(costs[i]-costs[selected[0]]),2),
                             p10_delta=round(float(dq[0]),2),p90_delta=round(float(dq[1]),2),
                             pit_windows=windows, window_definition="Within 1 modeled second of this sequence's optimum; not a confidence interval"))
    return output
