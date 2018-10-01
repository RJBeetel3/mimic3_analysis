try: 
	from setuptools import setup
except ImportError:
	from distutils.core import setup

requires = [
	'numpy==1.14.5',
	'pandas==0.23.0',
	'pluggy==0.6.0',
	'pytest==3.6.0',
	'python-dateutil==2.7.3',
	'scipy==1.1.0',
	'seaborn==0.8.1',
	'nose==1.3.8',
	'scikit-learn==0.20.0',
	'matplotlib==3.0.0',
	'PyYAML==3.13'
]

tests_require = [
	'pytest',
]

config = {
	'name': 'ICU Mortality Prediction',
	'version': '0.0',
	'description': 'ICU Mortality Prediction',
	'author': 'Rob Beetel',
	'author_email': 'beetel@gmail.com',
	'url': 'https://github.com/RJBeetel3/mimic3_analysis',
	'download_url': 'Not yet available online',
	'install_requires': requires,
						# ['nose', 'scikit-learn', 'scipy',
						#  'pandas', 'PyYAML'],
						#  'sys', 'os', 'nose', 'pandas',
						#  'datetime', 'numpy', 'dateutil',
						#  'sklearn', 'matplotlib', 'scipy',
						#  'PyYAML'],
	'packages': ['icu_mortality'],
	'scripts': [],
	'extras_require':  {
				'testing': tests_require
	},
	'tests_require': tests_require
}

setup(**config)
