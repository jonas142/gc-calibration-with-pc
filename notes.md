# BVC Model Integration

## Files to migrate
- BCsimulation
- parametersBC ?
- BCActivity
- HDCActivity
- Helper
- hdc_template (for init)
- HDCAttractorConnectivity
- network_numpy


- [x] integrate detection rays in environment
- [ ] plotting Classes/Functions
- [x] env.step should return angle of movement
    - angle is used as input for hdc network

### Open Questions
- Is the Head Direction Network actually used? Or is it computed elsewise?
    --> Yes it is used (confimr by following variable rates_hdc)
    - [x] update Files to migrate

## Communication between pc and bvc
- [x] add bvcactivity to pc.trackmovements
- [x] add bvcactivity to pc.init 
- [x] add bvcactivity to pc.firing_values
- [x] compute activity based on bvcactivity

# ACD integration

## Notes
- ecd and acd ring are added in hdc_template.py
- [x] find out what `rates_acd = list(hdc.getLayer('acd_ring'))` returns
    - could be Activity of the ACD cells and therefore exactly what I need :)
- Answer: It is exactly what we need.  

## Function explanations
### np.where(condition)
Example:
```
a = np.array([ 0, -1,  2,  3,  4,  3])
np.where(a == 3)
```
Returns indexes where condition is true.
