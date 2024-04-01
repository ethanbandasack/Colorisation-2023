# Colorisation-2023

L’association d'intelligence artificlle de CentraleSupélec, Automatants, a mis en place des projets de première année : un étudiant de deuxième année de l’association encadrait plusieurs nouveaux membres dans le cadre d’un premier projet pour découvrir l’IA.

Le projet auquel j’ai été affecté concernait la colorisation de mangas et de bandes-dessinées. En effet, les mangas deviennent de plus en plus populaires mais restent en noir et blanc. Il existe des versions colorisées pour certaines licences mais demandent du travail de la part de l’auteur ou de fans. Lorsqu’elles sont commercialisées, il s’agit d’un produit plébiscité. Le projet pourrait servir aux fans voulant avoir rapidement une version colorisée de leur série préférée. L’utilisation professionnelle de l’intelligence artificielle par des maisons d’édition et des auteurs eux-mêmes est aujourd’hui très controversée, mais elle reste possible : il n’est pas impossible que certaines maisons d’édition commandent de tels produits pour proposer une colorisation rapide et à bas coût.


Le projet repose sur une architecture de GAN (réseaux antagonistes) : d’une part un générateur, et d’autre part un discriminateur. Le générateur produit des images colorisées et le discriminateur juge si l’image est générée (fausse image colorisée) ou si elle est a été colorisée à la main (vraie image colorisée). Le générateur cherche à tromper le discriminateur qui lui cherche à ne pas se « faire avoir ». En s’améliorant, les colorisations deviennent de plus en plus crédibles. La génération passe par un réseau en U (U-net) : l’image en noir et blanc de base est décomposée en différentes composantes de plus petites tailles représentant chacune des informations précises, pour ensuite repasser sur une image aux dimensions classiques, mais colorisée.

Après s’être entraîné sur une grande base de données, le modèle est censé pouvoir coloriser d’autres images (étape de généralisation).

L’architecture GAN de base a du mal à généraliser à d’autres images, c’est pourquoi il est nécessaire d’améliorer le modèle : utilisation de ResNet, cycle GAN, ou utilisation de fonctions de coûts particulières.

A noter que le dataset n'est pas composé de couples image noir et blanc / image en couleurs pour l’entraînement, c’est pourquoi une image en noir et blanc était générée à partir de l’image en couleurs.

Les résultats obtenus sont assez mitigés : il y a une amélioration de l’image mais elle reste globalement en noir et blanc.
