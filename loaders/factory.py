from .base_loader import DocumentLoader
from .web_loader import WebsiteLoader
from .file_loader import FileLoader
import logging
from config import settings

logger = logging.getLogger(__name__)

class LoaderFactory:
    _loaders = {
        'web': WebsiteLoader,
        'file': FileLoader
    }
    
    @classmethod
    def create(cls, loader_type: str, **kwargs) -> DocumentLoader:
        loader_class = cls._loaders[loader_type]
        
        # Use config defaults if not specified
        if loader_type == 'web':
            chunk_size = kwargs.get('chunk_size', settings.WEB_CHUNK_SIZE)
        else:
            chunk_size = kwargs.get('chunk_size', settings.CHUNK_SIZE)
            
        return loader_class(chunk_size=chunk_size)