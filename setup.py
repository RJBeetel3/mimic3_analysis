try: 
	from setuptools import setup
except ImportError:
	from distutils.core import setup

requires = [
	'numpy==1.17.4',
	'pandas==0.25.3',
	'pluggy==0.13.1',
	'pytest==5.3.1',
	'python-dateutil==2.8.1',
	'scipy==1.3.3',
	'seaborn==0.9.0',
	'nose==1.3.7',
	'scikit-learn==0.21.3',
	'matplotlib==3.1.2',
	'PyYAML==5.1.2'
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
