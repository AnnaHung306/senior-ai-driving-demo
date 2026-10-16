"""Training GT only: simulator phase plus visible emissive-lens eligibility.

An actor's phase alone cannot label a back-facing or unresolvable lamp crop.
These fixed RGB tests only determine target observability, never inference.
"""
import numpy as np

LIGHT_STATES=['unknown_or_unobservable','red','yellow','green']


def phase_target(rgb,mask,simulator_phase,min_lens_pixels=2):
    pixels=rgb[mask].astype(np.int16)
    if not len(pixels): return {'light_state_class':0,'world_phase':simulator_phase,'emissive_pixels':{}}
    r,g,b=pixels.T
    counts={'Red':int(((r>=110)&(r-g>=45)&(r-b>=45)&(g<.65*r)).sum()),
            'Green':int(((g>=90)&(g-r>=30)&(g-b>=15)&(r<.65*g)).sum()),
            'Yellow':int(((r>=110)&(g>=90)&(b<.6*np.minimum(r,g))&(r>=.65*g)&(g>=.65*r)).sum())}
    phase=str(simulator_phase).split('.')[-1]
    value={'Red':1,'Yellow':2,'Green':3}.get(phase,0) if counts.get(phase,0)>=min_lens_pixels else 0
    return {'light_state_class':value,'world_phase':phase,'emissive_pixels':counts,
            'state_target_source':'simulator phase, conditional on >=2 visible semantic-lamp pixels with compatible emissive color; otherwise unknown',
            'limits':'Observability-conditioned synthetic GT; not human-certified state labels or all-face phase inference'}
