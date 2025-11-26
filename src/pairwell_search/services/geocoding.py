from geopy.geocoders import Nominatim
import difflib
import streamlit as st

def geocode_city(city_name, state_name=None):
    """
    Geocode a city name to get its latitude and longitude.
    Returns (latitude, longitude) or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="my_geocoder", timeout=5)
    location = geolocator.geocode({'city':city_name, 'state':state_name} if state_name else {'city':city_name})
    if location:
        return (location.latitude, location.longitude)
    else:
        return (None, None)
    
import difflib

def match_state_to_abbr(state):
    """Convert full state name (fuzzy) or abbreviation to its 2-letter abbreviation."""
    if not state or not isinstance(state, str):
        return None

    state = state.strip()

    # Already an abbreviation?
    if len(state) == 2 and state.isalpha():
        return state.upper()

    states = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
        'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
        'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
        'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
        'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
        'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
        'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND',
        'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
        'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD',
        'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
        'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
        'wisconsin': 'WI', 'wyoming': 'WY'
    }

    normalized = state.lower()

    # Try exact normalized match
    if normalized in states:
        return states[normalized]

    # Fuzzy match for mild misspellings
    close = difflib.get_close_matches(normalized, states.keys(), n=1, cutoff=0.6)
    print(close)
    if close:
        return states[close[0]]

    st.warning("No match found for state: {state}".format(state=state))
    return None
