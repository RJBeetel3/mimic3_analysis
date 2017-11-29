try: 
	from setuptools import setup
except ImportError:
	from distutils.core import setup
	
config = {
	'description': 'ICU Mortality Prediction', 
	'author': 'Rob Beetel', 
	'url': 'Not yet available online', 
	'download_url': 'Not yet available online', 
	'author_email': 'beetel@gmail.com', 
	'version': '0.1', 
	'install_requires': ['nose', 'scikit-learn', 'scipy', 
	                     'pandas', 'PyYAML'], 
	                     #'sys', 'os', 'nose', 'pandas', 
	                     #'datetime', 'numpy', 'dateutil', 
	                     #'sklearn', 'matplotlib', 'scipy', 
	                     #'PyYAML'], 
	'packages': ['icu_mortality'], 
	'scripts': [],
	'name': 'ICU Mortality Prediction'
	}

setup(**config)