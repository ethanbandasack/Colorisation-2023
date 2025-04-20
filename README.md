# Colorisation-2023

The artificial intelligence association of CentraleSupélec, Automatants, has set up first-year projects: a second-year student from the association supervised several new members as part of an introductory project to discover AI.

The project I was assigned to involved the colorization of manga and comics. Indeed, manga is becoming increasingly popular but remains in black and white. There are colored versions for some licenses, but they require work from the author or fans. When commercialized, it is a highly sought-after product. The project could be useful for fans who want a quick colored version of their favorite series. The professional use of artificial intelligence by publishers and authors themselves is highly controversial today, but it remains possible: it is not impossible that some publishers might commission such products to offer quick and low-cost colorization.

The project is based on a GAN (Generative Adversarial Network) architecture: a generator on one hand, and a discriminator on the other. The generator produces colored images, and the discriminator judges whether the image is generated (fake colored image) or hand-colored (real colored image). The generator tries to deceive the discriminator, which in turn tries not to be "fooled." As they improve, the colorizations become more and more credible. The generation process uses a U-net: the base black-and-white image is decomposed into various smaller components, each representing precise information, and then reconstructed into a standard-sized but colored image.

After training on a large dataset, the model is supposed to be able to colorize other images (generalization step).

The basic GAN architecture struggles to generalize to other images, which is why it is necessary to improve the model: using ResNet, cycle GAN, or specific cost functions.

Note that the dataset is not composed of black-and-white/colored image pairs for training, which is why a black-and-white image was generated from the colored image.

The results obtained are somewhat mixed: there is an improvement in the image, but it remains mostly black and white.
