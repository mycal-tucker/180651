from keras.layers import Layer
import keras.backend as K


class ProtoLayer(Layer):
    def __init__(self, num_prototypes, latent_dim, **kwargs):
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        super(ProtoLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.prototypes = self.add_weight(name='proto_kern',
                                          shape=(self.num_prototypes, self.latent_dim),
                                          initializer='uniform',
                                          trainable=True)
        super(ProtoLayer, self).build(input_shape)

    def call(self, feature_vectors, **kwargs):
        # Compute the distance between x and the protos
        features_squared = K.reshape(ProtoLayer.get_norms(feature_vectors), shape=(-1, 1))
        protos_squared = K.reshape(ProtoLayer.get_norms(self.prototypes), shape=(1, -1))
        dists_to_protos = features_squared + protos_squared - 2 * K.dot(feature_vectors, K.transpose(self.prototypes))

        alt_protos_squared = K.reshape(ProtoLayer.get_norms(self.prototypes), shape=(-1, 1))
        alt_features_squared = K.reshape(ProtoLayer.get_norms(feature_vectors), shape=(1, -1))
        dists_to_latents = alt_features_squared + alt_protos_squared - 2 * K.dot(self.prototypes, K.transpose(feature_vectors))
        return [dists_to_protos, dists_to_latents]

    @staticmethod
    def get_norms(x):
        return K.sum(K.pow(x, 2), axis=1)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.num_prototypes), (self.num_prototypes, input_shape[0])]
