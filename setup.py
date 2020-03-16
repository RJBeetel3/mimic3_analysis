try: 
	from setuptools import setup, find_packages
except ImportError:
	from distutils.core import setup

requires = [
	'numpy==1.18.1',
	'pandas==1.0.1',
	'pluggy==0.13.1',
	'pytest==5.4.1',
	'python-dateutil==2.8.1',
	'scipy==1.4.1',
	'seaborn==0.10.0',
	'nose==1.3.7',
	'scikit-learn==0.22.2',
	'matplotlib==3.2.0',
	'PyYAML==5.1.2'
]

tests_require = [
	'pytest',
]

config = {
	'name': 'icu_mortality',
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
	'packages': find_packages(),
	'scripts': [],
	'extras_require':  {
				'testing': tests_require
	},
	'tests_require': tests_require,
	'zip_safe': False
}

setup(**config)
