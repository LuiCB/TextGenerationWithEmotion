# Text Generation With Emotion

This is a team project. Emotional text is generated by a Variational AutoEncoder based model and a Seq-GAN based model.

### Variational AutoEncoder with Restrictions

we try to add some restrictions on the generative process and develop a version of variational auto-encoder with restrictions. Since our goal is to generate sentences with specified emotion types, 
we treat the emotion types as the restriction on our model.

To model this emotion
distribution, we simplify our model and dataset distribution
such that the level of positive and negative emotions follows
a normal distribution with the mean at 1 and -1 respectively.
Since this two distributions are symmetric, we just use one
latent variable Z to model it.


