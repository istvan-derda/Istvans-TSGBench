FROM docker.io/python:3.7.16

# Time-Transformer Dependencies
RUN pip install numpy==1.21.5
RUN pip install tensorflow==2.6.0
RUN pip install keras==2.6.0
RUN pip install scikit-learn==0.24.2
RUN pip install scipy==1.6.2
RUN pip install mgzip

WORKDIR /workspace

RUN git clone https://github.com/Lysarthas/Time-Transformer.git Time_Transformer
RUN cd Time_Transformer && git checkout ff8faa27353e721702aaa51136192f071e5a5e19

COPY run_time_transformer.py .

ENTRYPOINT python run_time_transformer.py


