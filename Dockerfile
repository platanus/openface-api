FROM bamos/openface

WORKDIR /app

RUN curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  && bunzip2 shape_predictor_68_face_landmarks.dat.bz2
RUN curl -O http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

ENV FLASK_APP face_api.py

# ENTRYPOINT ["./entrypoint.sh"]
CMD ./entrypoint.sh gunicorn -w 2 -b "0.0.0.0:$PORT" face_api:app

EXPOSE 5000
