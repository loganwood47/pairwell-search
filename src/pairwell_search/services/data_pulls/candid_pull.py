from ..apis.candid_api import CandidEssentialsAPI
from .. import db

import json

import os

# Used to fill database with nonprofits from Candid Essentials API

CANDID_ESSENTIALS_API_KEY = os.getenv("CANDID_ESSENTIALS_API_KEY")

candid = CandidEssentialsAPI(api_key=CANDID_ESSENTIALS_API_KEY)

# Example categories to seed
search_cats = [
    # 'animals',
    'native americans'
    # 'education', 
    # 'environment', 
    # 'climate', 
    # 'nature', 
    # 'wildlife', 
    # 'conservation', 
    # 'health', 
    # 'mental health',
    # 'hunger', 
    # 'poverty', 
    # 'housing', 
    # 'arts', 
    # 'culture', 
    # 'human rights', 
    # 'disaster relief', 
    # 'social justice',
    # 'youth development'
    # 'habitat restoration',
    # 'ocean conservation',
    # 'marine conservation',
    # 'watershed protection',
    # 'river conservation',
    # 'reforestation',
    # 'biodiversity',
    # 'sustainable agriculture',
    # 'civic engagement',
    # 'community development',
    # 'economic development',
    # 'workforce development',
    # 'financial literacy',
    # 'legal aid',
    # 'criminal justice reform',
    # 'immigrant services',
    # 'refugee assistance',
    # 'veteran affairs',
    # 'senior services',
    # 'disability rights',
    # 'women\'s empowerment',
    # 'LGBTQ+ services',
    # 'international development',
    # 'peacebuilding',
    # 'public policy',
    # 'scientific research',
    # 'medical research',
    # 'technology access',
    # 'historic preservation',
    # 'public broadcasting',
    # 'journalism',
    # 'consumer protection',
    # 'volunteerism',
    # 'land trust',
    # 'environmental justice',
    # 'environmental law',
    # 'park stewardship',
    # 'botanical gardens',
    # 'pollinator protection',
    # 'endangered species'
    ]

# search_cats = [ 
#     'Humanitarian aid',
#     'Medical research',
#     'Food security',
#     'Homeless services',
#     'Arts and culture',
#     'Human rights',
#     'Disaster relief',
#     'Mental health advocacy',
#     'Social justice',
#     'Youth development']


geoObj = { # Optional geo filter
    # "states": ["CA"],
    # "metros": ["Miami"]
}

result = candid._seed_nonprofits(queries=search_cats, geo_filter=geoObj, total_call_cap=240)


