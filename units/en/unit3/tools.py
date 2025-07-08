from langchain.tools import Tool
import random
import os
import requests
from huggingface_hub import list_models
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from units.en.unit3.retriever import bm25_retriever

def guest_info_retriever(query: str) -> str:
    """Retrieves guest information from database, with web search fallback for unfamiliar guests."""
    
    # First, try the local guest database
    results = bm25_retriever.invoke(query)
    
    if results and results[0].page_content.strip():
        # Found guest in local database
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        # Guest not found locally - search the web
        search_tool = DuckDuckGoSearchRun()
        web_query = f"{query} biography background information"
        
        try:
            web_results = search_tool.run(web_query)
            return f"Guest not found in local database. Here's what I found online:\n\n{web_results}"
        except Exception as e:
            return f"No matching guest information found in local database, and web search failed: {str(e)}"

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=guest_info_retriever,
    description="Retrieves detailed information about gala guests from local database, with web search fallback for unfamiliar guests."
)

def get_weather_info(city: str) -> str:
    """Fetches current weather information for a specified city using OpenWeatherMap API.
    Args:
        city: Name of the city (e.g., 'London')
    """
    api_key = "aa5af88a9d9234e7ff85d955f024cc67"
    print(f"Using API key: {api_key}")  # Debugging line
    if not api_key:
        return "API key not found. Make sure OPENWEATHERMAP_API_KEY is set in your environment variables."

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(url, params=params)
        data = response.json()
        print(f"Weather API response: {data}")  # Debugging line
        if response.status_code != 200:
            return f"Error: {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]
        return f"Current weather in {city}: {temp}°C, {condition}."

    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return f"An error occurred while fetching weather data: {str(e)}"

# Initialize the weather tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches current weather information for a specified city using OpenWeatherMap API."
)
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)
