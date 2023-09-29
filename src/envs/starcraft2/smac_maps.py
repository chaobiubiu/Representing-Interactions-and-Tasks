from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib
from smac.env.starcraft2.maps import smac_maps

map_param_registry = {
    # This full_map is only used to get the unit_type (ID) of all units in starcraft 2.4.10.
    "full_map_two":{
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 9,
        "map_type": "full_classes"
    },
    "1o_10b_vs_1r": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_2r_vs_4r": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },
        "bane_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    }
}


smac_maps.map_param_registry.update(map_param_registry)

def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]

custom_maps = ["full_map_template", "empty_passive", "empty_aggressive", "terran_vs_terran", "5m_vs_6m_alt", "5m_vs_6m_alt1xtra"]

for name in list(map_param_registry.keys()) + custom_maps:
    globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))

# for name in map_param_registry.keys():
#     globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))