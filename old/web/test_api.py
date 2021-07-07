import json
from random import randint, sample


COUNTRIES = ['BLR', 'DN', 'FI', 'NO', 'RUS', 'SW', 'UA', 'UK', 'USA']

action = {
    'campaign_id': 1283,
    'ad_id': 283,
    'segment': 'poor',
    'build_id': sample(['A', 'B'], 1)[0],  # A or B
    'items': {
        'time_spent': 324,  # in sec
        'time_hover': 21,
        'time_session': 1894
    },
    'user': {
        'ip': '{}.{}.{}.{}'.format(*sample(range(0,255),4)),
        'age': randint(14, 60),
        'country': sample(COUNTRIES, 1)[0]
    }
}

# create file with some user data
fw = open('../data/actions.json', 'w', encoding='utf-8')
json.dump(action, fw)
fw.close()


file = open('../data/actions.json', 'r+', encoding='utf-8')
acts = file.read()
actions = [] if acts == '' else [json.loads(acts)]
actions.append(action)
file.truncate(0)
json.dump(actions, file, indent=2)
file.close()

