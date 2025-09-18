from geopy.geocoders import Nominatim

def geocode_city(city_name, state_name=None):
    """
    Geocode a city name to get its latitude and longitude.
    Returns (latitude, longitude) or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode({'city':city_name, 'state':state_name} if state_name else {'city':city_name})
    if location:
        return (location.latitude, location.longitude)
    else:
        return (None, None)