FROM python:3.7.6
RUN useradd -u 1000 docker_user
RUN chown -R docker_user:docker_user /home/
USER docker_user
RUN pip install spacy==2.2.4
#RUN pip install notebook==6.5.4 ipywidgets==8.0.7
#RUN pip install de-core-news-md@https://github.com/explosion/spacy-models/releases/download/de_core_news_md-2.2.5/de_core_news_md-2.2.5.tar.gz#sha256:f325eebb70e0c4fe280deb605004667a65bb3457dcde7a719926a4e040266cca
#CMD ["jupyter", "notebook", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "/veld/executable/src"]

