from setuptools import setup

setup(
    name='gym-ur5_gripper',
    version='0.0.1',
    description='A gym mujoco environment of ur5 3 finger gripper',
    author='Cong Wang',
    author_email='wangcongrobot@gmail.com',
    install_requires=['gym==0.14.0', 
                      'numpy',
                      'mujoco_py==2.0.2.2']
)