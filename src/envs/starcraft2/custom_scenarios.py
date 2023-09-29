from numpy.random import RandomState
from os.path import dirname, join
from functools import partial
from itertools import combinations_with_replacement, product


def get_all_unique_teams(all_types, min_len, max_len):
    all_uniq = []
    for i in range(min_len, max_len + 1):
        all_uniq += list(combinations_with_replacement(all_types, i))
    all_uniq_counts = []
    for scen in all_uniq:
        curr_uniq = list(set(scen))
        uniq_counts = list(zip([scen.count(u) for u in curr_uniq], curr_uniq))
        all_uniq_counts.append(uniq_counts)
    return all_uniq_counts


def symmetric_armies(army_spec, ally_centered=False,
                     rotate=False, separation=10,
                     jitter=0, episode_limit=100, map_name="empty_passive",
                     n_extra_tags=0,
                     rs=None):
    if rs is None:
        rs = RandomState()

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0],
                                                     n_unit_range[1]))
    unique_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]

    scenarios = list(zip(unique_teams, unique_teams))
    # sort by number of types and total number of units
    max_types_and_units_team = sorted(unique_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_team,
                                    max_types_and_units_team)

    num_scenarios = len(scenarios)
    threshold = int(num_scenarios * 0.8)
    training_scenarios = scenarios[:threshold]
    testing_scenarios = scenarios[threshold:]

    scenario_dict = {'scenarios': scenarios,
                     'training_scenarios': training_scenarios,
                     'testing_scenarios': testing_scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'n_extra_tags': n_extra_tags,
                     'map_name': map_name}
    return scenario_dict


def asymm_armies(army_spec, spec_delta, ally_centered=False,
                 rotate=False, separation=10,
                 jitter=0, episode_limit=100, map_name="empty_passive",
                 n_extra_tags=0,
                 rs=None):
    if rs is None:
        rs = RandomState()

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0],
                                                     n_unit_range[1]))
    enemy_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]
    agent_teams = [[(max(num + spec_delta.get(typ, 0), 0), typ) for num, typ in team] for team in enemy_teams]

    scenarios = list(zip(agent_teams, enemy_teams))
    # sort by number of types and total number of units
    max_types_and_units_ag_team = sorted(agent_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_en_team = sorted(enemy_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_ag_team,
                                    max_types_and_units_en_team)

    num_scenarios = len(scenarios)
    threshold = int(num_scenarios * 0.8)
    training_scenarios = scenarios[:threshold]
    testing_scenarios = scenarios[threshold:]

    scenario_dict = {'scenarios': scenarios,
                     'training_scenarios': training_scenarios,
                     'testing_scenarios': testing_scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'n_extra_tags': n_extra_tags,
                     'map_name': map_name}
    return scenario_dict


"""
The function in the registry needs to return a tuple of two lists, one for the
ally army and one for the enemy.
Each is of the form [(number, unit_type, pos), ....], where pos is the starting
positiong (relative to center of map) for the corresponding units.
The function will be called on each episode start.
Currently, we only support the same number of agents and enemies each episode.
"""
# empty_passive
custom_scenario_registry = {
    # 6 maps
    "3-8m_symmetric": partial(symmetric_armies,
                              [(('Marine',), (3, 8))],
                              rotate=True,
                              ally_centered=False,
                              separation=14,
                              jitter=1, episode_limit=100, map_name="full_map_template"),
    # 6 maps
    "6-11m_mandown": partial(asymm_armies,
                             [(('Marine',), (6, 11))],
                             {'Marine': -1},
                             rotate=True,
                             ally_centered=False,
                             separation=14,
                             jitter=1, episode_limit=100, map_name="full_map_template"),
    # 4+5+6+7+8+9=39 maps
    "3-8sz_symmetric": partial(symmetric_armies,
                               [(('Stalker', 'Zealot'), (3, 8))],
                               separation=14,
                               jitter=1, episode_limit=150, map_name="full_map_template"),
    # (4+5+6+7)x3=66 maps
    "3-8MMM_symmetric": partial(symmetric_armies,
                                [(('Marine', 'Marauder'), (3, 6)),
                                 (('Medivac',), (0, 2))],
                                rotate=True,
                                ally_centered=False,
                                separation=14,
                                jitter=1, episode_limit=150, map_name="full_map_template"),
    # (4+5+6+7)x3=66 maps
    "3-8csz_symmetric": partial(symmetric_armies,
                                [(('Stalker', 'Zealot'), (3, 6)),
                                 (('Colossus',), (0, 2))],
                                rotate=True,
                                ally_centered=False,
                                separation=14,
                                jitter=1, episode_limit=150, map_name="full_map_template"),
}

if __name__ == '__main__':
    symmetric_armies([(('Marine',), (3, 8))],
                              rotate=True,
                              ally_centered=False,
                              separation=14,
                              jitter=1, episode_limit=100, map_name="empty_passive")        # Empty_passive may result in errors.
