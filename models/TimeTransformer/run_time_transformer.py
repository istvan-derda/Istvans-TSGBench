import pickle
import mgzip
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import os


from Time_Transformer.aae import aae_model
from Time_Transformer.networks import timesformer_dec, cnn_enc, discriminator

data_root = '../../data/'

def main():
    train_data = load_train_data()

    gen_data = generate_from(train_data)

    persist_gen_data(gen_data)


def load_train_data():
    dataset_name = get_dataset_name()
    with mgzip.open(f'{data_root}ori/{dataset_name}/{dataset_name}_train.pkl') as f:
        train_data = pickle.load(f)
    return train_data


def persist_gen_data(data):
    dataset_name = get_dataset_name()
    dir_path = f'{data_root}gen/{dataset_name}/'
    file_path = dir_path+f'{dataset_name}_gen.pkl'
    os.makedirs(dir_path, exist_ok=True)
    with mgzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)


def get_dataset_name():
    return os.environ['TSGB_USE_DATASET']


def generate_from(train_data):
    input_shape = train_data.shape[1:]
    latent_dim = 16

    model = compile_model(
        latent_dim=latent_dim, 
        input_shape=input_shape
    )
    
    # from Time-Transformer/tutorial.ipynb #Train model
    model.fit(train_data, epochs=20, batch_size=128)

    z = tf.random.normal([train_data.shape[0], latent_dim], 0.0, 1.0)
    gen_data = model.dec.predict(z)

    return gen_data


def compile_model(input_shape, latent_dim):
    # from Time-Transformer/tutorial.ipynb #Build model
    enc = cnn_enc(
        input_shape=input_shape,
        latent_dim=latent_dim,
        n_filters=[64, 128, 256],
        k_size=4,
        dropout=0.2
    )

    dec = timesformer_dec(
        input_shape=latent_dim,
        ts_shape=input_shape,
        head_size=64,
        num_heads=3,
        n_filters=[128, 64],
        k_size=4,
        dilations=[1,4],
        dropout=0.2
    )

    disc = discriminator(input_shape=latent_dim, hidden_unit=32)

    def ae_loss(ori_ts, rec_ts):
        return tf.keras.metrics.mse(ori_ts, rec_ts)

    def dis_loss(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)

    def gen_loss(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)

    ae_schedule = PolynomialDecay(initial_learning_rate=0.005, decay_steps=300, end_learning_rate=0.0001, power=0.5)
    dc_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=300, end_learning_rate=0.0001, power=0.5)
    ge_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=300, end_learning_rate=0.0001, power=0.5)

    ae_opt = tf.keras.optimizers.Adam(ae_schedule)
    dc_opt = tf.keras.optimizers.Adam(dc_schedule)
    ge_opt = tf.keras.optimizers.Adam(ge_schedule)

    model = aae_model(
        encoder=enc, 
        decoder=dec, 
        discriminator=disc, 
        latent_dim=latent_dim, 
        dis_steps=1,
        gen_steps=1)
    
    model.compile(rec_opt=ae_opt, rec_obj=ae_loss, dis_opt=dc_opt, dis_obj=dis_loss, gen_opt=ge_opt, gen_obj=gen_loss)

    return model


if __name__ == '__main__':
    main()