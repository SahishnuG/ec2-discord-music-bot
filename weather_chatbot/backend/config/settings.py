from dotenv import load_dotenv,find_dotenv
import os
ENV_PATH = find_dotenv(".env") #find path cause its in config and env is in root
load_dotenv(ENV_PATH)

class Settings:
    """Settings class to  handle all the data"""
    # Read from env to initialize vars
    def __init__(self) -> None:
        self.__langfuse_host=os.getenv("LANGFUSE_HOST") #__ for name mangling so we don't accidentaly access var instead of method
        self.__langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY")
        self.__langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY")
        self.__weather_api_key = os.getenv('WEATHER_API_KEY')
        self.__router_system = os.getenv("ROUTER_SYSTEM_PROMPT", '""') #initially empty, fill when main is run, access after new obj made in api
        self.__summariser_system = os.getenv("SUMMARISER_SYSTEM_PROMPT", '""') #initially empty, fill when main is run, access after new obj made in api
    
    # Functions to access vars
    @property #to access method like its a var (no ()). Also makes it read only
    def langfuse_host(self):
        return self.__langfuse_host
    @property
    def langfuse_public_key(self):
        return self.__langfuse_public_key
    @property
    def langfuse_secret_key(self):
        return self.__langfuse_secret_key
    @property
    def weather_api_key(self):
        return self.__weather_api_key
    @property
    def router_system(self):
        return self.__router_system
    @property
    def summariser_system(self):
        return self.__summariser_system
    @property
    def env_path(self):
        return ENV_PATH
    
    