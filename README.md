# RASA_Edubot

### Creation of Env
```
conda create --name RasaEnvFinal python==3.7
conda activate RasaEnvFinal
```

### How to install Rasa

```
conda install tensorflow==2.1.0
pip install rasa[spacy]==1.8.0
python -m spacy download en_core_web_md
```
You can skip the first step in linux sometimes if it is showing an already existing unoverwritable error


Note: The above step requires Python 3.7 or 3.8
If there are some spacy errors try to upgrade your pip and run again.

SOmetimes your sdk version and core verisons are different. So in such cases just reinstall the sdk to the correct core version using the below command.

```
pip install rasa-sdk~=1.8.0
```
