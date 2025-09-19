import csv
import random
import json
from faker import Faker

fake = Faker()
Faker.seed(47)
random.seed(47)

N_USERS = 10000

# Buckets for controlled variety
interest_pool = [
    'animals',
    'education', 
    'environment', 
    'climate', 
    'nature', 
    'wildlife', 
    'conservation', 
    'health', 
    'mental health',
    'hunger', 
    'poverty', 
    'housing', 
    'arts', 
    'culture', 
    'human rights', 
    'disaster relief', 
    'social justice',
    'youth development'
    'habitat restoration',
    'ocean conservation',
    'marine conservation',
    'watershed protection',
    'river conservation',
    'reforestation',
    'biodiversity',
    'sustainable agriculture',
    'civic engagement',
    'community development',
    'economic development',
    'workforce development',
    'financial literacy',
    'legal aid',
    'criminal justice reform',
    'immigrant services',
    'refugee assistance',
    'veteran affairs',
    'senior services',
    'disability rights',
    'women\'s empowerment',
    'LGBTQ+ services',
    'international development',
    'peacebuilding',
    'public policy',
    'scientific research',
    'medical research',
    'technology access',
    'historic preservation',
    'public broadcasting',
    'journalism',
    'consumer protection',
    'volunteerism',
    'land trust',
    'environmental justice',
    'environmental law',
    'park stewardship',
    'botanical gardens',
    'pollinator protection',
    'endangered species'
]

engagement_pool = ["donate", "volunteer", "fundraise", "advocacy"]

states = ["CA", "NY", "TX", "FL", "IL", "WA", "MA", "GA", "NC", "OH"]

first_name_pool = [fake.first_name() for _ in range(N_USERS)]
last_name_pool = [fake.last_name() for _ in range(N_USERS)]

def random_income():
    return random.choice([25000, 50000, 75000, 100000, 150000, 250000])

def random_budget(income):
    # donation budget 1-5% of income
    return int(income * random.uniform(0.01, 0.05))

def random_interests():
    return json.dumps(random.sample(interest_pool, random.randint(1, 3)))

def random_engagement():
    return json.dumps(random.sample(engagement_pool, random.randint(1, 2)))

with open("users.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "created_at", "city", "state", "income",
        "interests", "donation_budget", "engagement_prefs", "first_name", "last_name"
    ])
    for i in range(1, N_USERS + 1):
        income = random_income()
        row = [
            i,
            fake.date_time_this_decade().isoformat(),
            fake.city(),
            random.choice(states),
            income,
            random_interests(),
            random_budget(income),
            random_engagement(),
            first_name_pool[i-1],
            last_name_pool[i-1]
        ]
        writer.writerow(row)

print("Generated users.csv with", N_USERS, "rows")
