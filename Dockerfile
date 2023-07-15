FROM python:3.7.6
RUN useradd -u 1000 docker_user
RUN chown -R docker_user:docker_user /home/
USER docker_user
RUN pip install spacy==2.2.4
RUN pip install de-core-news-md@https://github.com/explosion/spacy-models/releases/download/de_core_news_md-2.2.5/de_core_news_md-2.2.5.tar.gz#sha256:f325eebb70e0c4fe280deb605004667a65bb3457dcde7a719926a4e040266cca
#RUN pip install spacy==3.6.0
#RUN pip install notebook==6.5.4
#RUN pip install de_dep_news_trf@https://github.com/explosion/spacy-models/releases/download/de_dep_news_trf-3.5.0/de_dep_news_trf-3.5.0.tar.gz
#RUN pip install ipywidgets==8.0.7
#CMD ["python", "/veld/executable/src/converters.py"]
#CMD ["jupyter", "notebook", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "/veld/executable/src"]

