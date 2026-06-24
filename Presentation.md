---
marp: true
paginate: true
math: mathjax

style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }



  section.titre {
    background-color: #FAFAFA;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start; /* Alignement à gauche très élégant */
    padding: 80px;
  }
  section.titre h1 {
    font-size: 2em;
    border-bottom: none;
    padding-bottom: 0;
    margin-bottom: 40px;
    font-weight: 700;
    line-height: 1.2;
  }
  section.titre p {
    color: #4B5563;
    font-size: 0.9em;
    margin: 6px 0;
  }
  section.titre .meta-jury {
    margin-top: 20px;
    color: #6B7280;
    font-size: 0.8em;
    padding-top: 20px;
    width: 100%;
  }

  img {
    object-fit: contain;
    display: block;
    margin: 0 auto;
  }
  img.emoji {
  display: inline;
  margin: 0 0.15em;
  vertical-align: -0.1em;
  }

    h1 {
    font-size: 1.45em;
    font-weight: 700;
    color: #1A3354;
    margin-top: 12px;
    margin-bottom: 32px;
    padding-bottom: 14px;
    border-bottom: 1.5px solid #E2E8F0;
    letter-spacing: -0.02em;
    line-height: 1.25;
  }

  h2 {
    font-size: 1.1em;
    font-weight: 600;
    color: #1A3354;
    margin-bottom: 12px;
  }

  section:not(.titre) ul {
    list-style: none;
    padding-left: 0.4em;
  }

  section:not(.titre) ul li {
    margin-bottom: 0.55em;
    padding-left: 1.3em;
    position: relative;
  }

  section:not(.titre) ul li::before {
    content: '–';
    position: absolute;
    left: 0;
    color: #3B7DD8;
    font-weight: 700;
  }

    /* ── Captions (paragraphe ne contenant qu'un em) ── */
  p:has(> em:only-child) {
    text-align: center;
    color: #718096;
    font-size: 0.9em;
    margin-top: 8px;
  }

  /* ── Numérotation ── */
  section::after {
    font-size: 0.62em;
    color: #A0AEC0;
    font-weight: 400;
    bottom: 22px;
    right: 72px;
  }


    /* ════════════════════════════════════════════
     SLIDE DE SECTION (séparateur)
  ════════════════════════════════════════════ */

  section.sep {
    background: #3B7DD8;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 80px;
  }

  section.sep::before { display: none; }
  section.sep::after  { display: none; }

  section.sep h1 {
    color: #FFFFFF;
    font-size: 2em;
    font-weight: 700;
    letter-spacing: -0.025em;
    border-top: 1.5px solid rgba(255, 255, 255, 0.30);
    border-bottom: 1.5px solid rgba(255, 255, 255, 0.30);
    border-left: none;
    padding: 22px 40px;
    margin: 0;
    line-height: 1.3;
  }


    /* Les boites*/
    .box {
    padding: 14px 20px;
    border-radius: 6px;
    margin: 16px 0;
    font-size: 0.88em;
    line-height: 1.6;
    }
    .box.info   { background: #EFF6FF; border-left: 4px solid #3B7DD8; color: #1E3A5F; }
    .box.warn   { background: #FFFBEB; border-left: 4px solid #F59E0B; color: #78350F; }
    .box.danger { background: #FEF2F2; border-left: 4px solid #EF4444; color: #7F1D1D; }
    .box.tip    { background: #F0FDF4; border-left: 4px solid #22C55E; color: #14532D; }


    .source {
    position: absolute;
    bottom: 44px;
    right: 72px;
    font-size: 0.65em;
    color: #97a1ad  /* pour fond sombre */
    }

    blockquote {
    border-top: 0.1em dashed #555;
    font-size: 60%;
    margin-top: auto;
    } 


     strong, b {
    color: #3B7DD8;
  }


---

<!-- _class: titre -->

# **A Hardware-Agnostic GPU Acceleration for the n-Dimensional QuickHull Algorithm**


<p><strong>Alexis ENGLEBERT</strong></p>
<p>Sous la direction du Pr. Benoît LEGAT</p>

<div class="meta-jury">
  Jury : Jean-François REMACLE, Nathan TIHON, Tom BARBETTE <br>
  24 juin 2026 - Université catholique de Louvain (UCLouvain)
</div>

---

# Table des matières

- Une enveloppe convexe ?
- QuickHull
- État de l'art
- Contribution
- Résultats
- Conclusion

--- 
<!-- _class: sep -->

# Une enveloppe convexe ?

---

# Une enveloppe convexe ?

<div class="columns">

<div>

Une enveloppe convexe d'un ensemble $\mathcal{S}$ est le plus petit ensemble convexe qui contient $\mathcal{S}$.

<div class="box info">

$\rightarrow$ dimension $n$ !

</div>

</div>

<div>

![w:800px](./figures/convexhul_2d_exemple.png)

*Figure 1: Exemple d'enveloppe convexe en 2D*
</div>

--- 

# Pourquoi ?

<div class="columns">
<div>

* Jeu vidéo $\rightarrow$ collisions
* Machine learning $\rightarrow$ boundary
* Computer vision $\rightarrow$ Analyse de silhouettes
* $\cdots$
</div>

<div>

![alt text](./figures/dragon.png)

*Figure 2: Enveloppe convexe d'un modèle 3D contenant 3.6 millions de points [1]*
</div>

--- 
# Motivation

- Les données grandissent de plus en plus et les GPU offrent un parallélisme massif $\rightarrow$ rendre le calcul plus rapide !
- Mais il n'existe à ma connaissance aucune bibliothèque pour calculer l'enveloppe convexe sur GPU qui soit :
  - n-D
  - hardware-agnostique
  - open source

$\rightarrow$ **Objectif**: Créer une bibliothèque open source qui implémente n-D QuickHull sur GPU. 

---
# Les algorithmes existants 
- Gift wrapping
- Double description
- Reverse search
- $\cdots$
- **QuickHull**

---
<!-- _class: sep -->

# QuickHull

---

# QuickHull

Étape 1: Trouver un simplexe.
Étape 2: Ajouter le point le plus loin de l'enveloppe.
Étape 3: Répéter l'étape 2, jusqu'à ce qu'il n'y ait plus de point.

---
![bg](./figures/simplex_0.png)

<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>

--- 
![bg](./figures/simplex_1.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>


--- 
![bg](./figures/simplex_2.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>


--- 
![bg](./figures/simplex_3.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>

--- 
![bg](./figures/simplex_4.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>

--- 
![bg](./figures/simplex_5.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>

--- 
![bg](./figures/simplex_6.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>

--- 
![bg](./figures/simplex_7.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>

--- 
![bg](./figures/simplex_8.png)
<div class="source">Source : D. Gregorius (Valve), Implementing QuickHull, GDC 2014</div>


---
# QuickHull: Opérations à virgule flottantes 

<div class="columns">
<div>

* Les calculs à virgule flottantes sont sujets à des instabilités dues à leur représentation finie. (IEEE 754)
<br>
$$

0.1 + 0.2 = 0.30000000000000004
$$

</div>


<div>

![h:400](./figures/xkcd.png)

*Source: https://www.explainxkcd.com/wiki/index.php/3228:_Day_Counter*

</div>

---
# QuickHull: Opérations à virgule flottantes 

<div class="columns">
<div>

## Epsilon ($\epsilon$)

* On ajoute une valeur de sécurité ignorant les points qui sont trop proches d'une face.
* Permet d'éviter les divisions par 0.

</div>

<div>

## Jittering

* Ajoute une très petite valeur aux coordonnées des points pour éviter que les points soient trop proches.

</div>

</div>


---
# Complexité

- L'algorithme de QuickHull a une complexité **moyenne** de $\mathcal{O}(n \log v)$.

- Dans le pire des cas $\mathcal{O}(n^{\lfloor{d/2}\rfloor})$ $\rightarrow$ quand $d$ augmente, le temps d'exécution augmente fortement :turtle:

- QuickHull implémenté dans la bibliothèque qhull $\rightarrow$ bibliothèque de référence.

---
<!-- _class: sep -->

# État de l'art

---

# État de l'art: Tzeng & Owens

* Implémentation de QuickHull en 2D.
* Se base principalement sur le scan segmenté.
* **Speedup**:  ~x23 par rapport à *qhull*.
---

# État de l'art: Tang et al.

* Implémentation de QuickHull en 3D (extensible en n-D).
* Création d'une enveloppe pas spécialement convexe de manière greedy
* Filtrage des données sur GPU
* Lance *qhull* pour le reste des points.
* **Speedup**: ~40x par rapport à *qhull* 

---
# État de l'art: ghull

* Implémentation de QuickHull en 3D.
- Approxime l'enveloppe convexe via un diagramme de Voronoi
- Ajoute les points en dehors de l'enveloppe un à un.
- **Speedup**: ~6x par rapport à *qhull*.

---

# État de l'art: CudaHull
* Implémentation de QuickHull en 3D.
* Filtre d'abord les données
* Implémentation de QuickHull sur GPU.
* **Speedup**: ~39x par rapport à *qhull*.

---

# État de l'art: Récapitulatif

| Algorithme      | Dimensions | Backend  | Open-source |
|----------------|------------|-----------|-------------|
| Tzeng & Owens  | 2D         | CUDA      | Non         |
| Tang et al.    | 3D+        | CUDA      | Non         |
| ghull          | 3D         | CUDA      | Oui         |
| CudaHull       | 3D         | CUDA      | Non         |


<span>**Problème** $\rightarrow$ uniquement CUDA et 2D/3D :cry: </span>

---
<!-- _class: sep -->

# Contribution
---

# Contribution
**GPUConvexHull.jl**: Implémentation de n-D QuickHull sur GPU en Julia. 
-

$\rightarrow$ Se base sur l'aspect segmenté des données.
$\rightarrow$ Se base fortement sur des primitives GPU.
$\rightarrow$ Implémenté avec la bibliothèque *KernelAbstractions.jl* 

---

# Primitives GPU
- **Réduction Min-Max**: calcule le min et le max d'une liste.
- **Scan segmenté**: effectue un préfix sum sur des segments d'une liste.
- **Compact**: Enlève les données d'une liste. 

---

# Création du simplexe


<div class="columns">
<div>

*  Prendre le min et le max dans chaque dimension (GPU)
* **Failsafe:** S'il n'y a pas $d+1$ points alors on fait comme QuickHull (CPU).

$\rightarrow$ On retire les points à l'intérieur avec un compact (GPU)
</div>
<div>

![alt text](figures/image-4.png)
</div>

---

# Assigner les points aux faces
<div class="columns">
<div>

* Pour chaque point on va calculer quelle face est la plus proche et lui assigner cette face (GPU)

* Permet de faire des opérations par faces.

$\rightarrow$ Nous permet de trouver le point le plus loin pour chaque face.

</div>

<div>

![alt text](figures/image-5.png)

</div>

---

# Mettre à jour notre enveloppe convexe

<div class="columns">
<div>

* Ajouter un point $\rightarrow$ change la topologie $\rightarrow$ recalculer les hyperplans.

- Difficile de paralléliser sur GPU de manière efficace $\rightarrow$ implémentation sur CPU pour l'instant. 

</div>

<div> 

![h:400](figures/image-6.png)

*Source: D. Gregorius (Valve), Implementing QuickHull, GDC 2014*

</div>

---

# Architecture du programme
![](./figures/Création%20du%20simplexe.png)

*Figure 3: Architecture du programme. Les cases vertes tournent sur le GPU, les rouges sur le CPU*

---

<!-- _class: sep 
_paginate: false-->

# Résultats

---
# Résultats: Temps d'exécution hypercube
![w:800px](./figures/execution_times_subplots.png)

---
# Résultats: Temps d'exécution hypersphère

![](./figures/sphere_3D_4D.png)

---

# Résultats: La mémoire ? (1) 
![](./figures/gpu_allocations_sphere.png)


---

# Résultats: La mémoire ? (2)

![alt](figures/image.png)

---

# Résultats: Qu'est-ce qui prend du temps ? (1) 

![w:900](figures/image-7.png)


*Figure 4: Distribution unsiforme dans un cube en 3D (N = $10^6$ ).* 


---
# Résultats: Qu'est-ce qui prend du temps ? (2)

![w:900](figures/image-8.png)

*Figure 5: Distribution uniforme sur une sphere en 3D (N = $10^6$ ).* 

---
# Résultats: Bottleneck

- Le scan segmenté.
- Les allocations de mémoire
- L'insertion des points dans l'enveloppe convexe.
$\rightarrow$ avec $10^5$ points (0.8 MB) sur une sphère, **uniquement** l'insertion de point alloue ~18,392MB

---

# Validation

**Problème :** Qhull fusionne les faces coplanaires (face merging), 
Pas nous $\rightarrow$ on ne peut pas comparer les sommets directement.

**Notre approche :** vérifier que nos sommets se trouvent 
sur les hyperplans générés par Qhull.

<div style="display: flex; justify-content: center; align-items: center;">

| Dimension | N | Distribution | Écart max |
|:---|---|---|---|
| 2D | $100$ | Uniforme | $1.39 × 10^ {-17}$ |
| 3D | $1000$ | Uniforme | $2.22 × 10^ {-16}$ |
| 3D | $1000$ | Sphère | $2.22 × 10^ {-16}$ |
| 4D | $500$ | Uniforme | $4.44 × 10^ {-16}$ |

</div>


---

# Pistes d'amélioration

- **Face merging**: éviter les instabilités sur inputs dégénérés

- **Topologie entièrement sur GPU**:  bottleneck en haute dimension
  *(en cours de développement)*.

- **Réduire les allocations dynamiques**: principale cause 
  du ralentissement en haute dimension.

- **Epsilon adaptatif** : actuellement fixé à $10^{-9}$ arbitrairement,
  Qhull l'adapte aux données.

- **Scan plus efficace**: algorithmes comme Merrill & Garland 
  potentiellement 2x plus rapides.
---

# Conclusion

* **GPUConvexHull.jl**: première bibliothèque open-source, 
hardware-agnostique et n-dimensionnelle pour QuickHull sur GPU.

* Speedup jusqu'à **~20×** par rapport à *qhull* en 2D/3D.

* Bonne base pour la suite. 

---
<!-- _class: sep 
_paginate: false-->

# Questions ?

---

# Références

[1] Mingcen Gao et al. “gHull: A GPU algorithm for 3D convex hull”. In: ACM
Transactions on Mathematical Software (TOMS) 40.1 (2013), pp. 1–19.


