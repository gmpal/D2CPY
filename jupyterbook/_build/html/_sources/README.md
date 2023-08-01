# Welcome to D2CPY!

`D2CPY` is a Python library that serves as a Python adaptation of the R package [`D2C`](https://github.com/gbonte/D2C). Originally developed in R, `D2C` is a powerful tool based on the research paper {cite}`bontempi15a` titled "From Dependency to Causality: A Machine Learning Approach" by Gianluca Bontempi and Maxime Flauder, published in the Journal of Machine Learning Research (JMLR) in 2015. The paper itself can be found at [this link](http://jmlr.org/papers/v16/bontempi15a.html).

The research paper addresses the fundamental relationship between statistical dependency and causality, which forms the basis of various statistical approaches to causal inference. One of the key contributions of the paper is the demonstration that causal directionality can be accurately inferred even in Markov indistinguishable configurations through data-driven approaches.

`D2C` implements a supervised machine learning approach that enables the detection of directed causal links between variables in multivariate settings where the number of variables exceeds two (n>2). The methodology leverages the asymmetry of conditional (in)dependence relations among the members of the Markov blankets of causally connected variables. By utilizing statistical descriptors with asymmetry, `D2C` demonstrates that supervised learning techniques can effectively extract valuable causal information from multivariate distributions where n>2.

With `D2CPY`, Python developers gain access to the rich functionalities and capabilities originally offered by the `D2C` package. It empowers users to explore causal relationships in complex multivariate systems and facilitates the extraction of causal insights from data using state-of-the-art machine learning techniques.

By providing a Python implementation of the D2C package and incorporating the advancements presented in the research paper, `D2CPY` offers a seamless and efficient experience for causal inference and analysis in Python, expanding the reach of this significant research to a wider community of Python developers and data scientists.

Start leveraging the power of causal inference in your Python projects with `D2CPY` today.
