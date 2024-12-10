from setuptools import setup, find_packages

setup(
    name="FitWizard",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A fitness tracking application using pose estimation.",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python>=4.5.3',
        'mediapipe>=0.8.9',
        'pyttsx3>=2.90',
        'numpy>=1.21.0',
        'PyQt5>=5.15.4',
    ],
    entry_points={
        'console_scripts': [
            'fitwizard = gui.main_window:main',  # Updated to match the location of the main function
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)