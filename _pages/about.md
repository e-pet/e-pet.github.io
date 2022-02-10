---
permalink: /
title: ""
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

**Hi there, I'm Eike.**

I am a postdoctoral researcher at [DTU Compute](https://www.compute.dtu.dk/english), in the section for [Visual Computing](https://www.compute.dtu.dk/english/research/Research-sections/Visual-Computing).
In my current research within the project [Bias and Fairness in Medicine](http://fairmed.compute.dtu.dk/), I (together with my amazing [collaborators](http://fairmed.compute.dtu.dk/team.php)) attempt to answer the following questions: 

> What does it mean for a data-driven health risk score model to be "fair"? How can we test whether a model *is* fair, and how can we *design* such a model?

Allow me to elaborate just a tiny bit. Health risk score models are likely to be used for *resource prioritization* in the healthcare system, for example by influencing who gets access to acute care and who does not.[^1] As these are decisions that influence human livelihoods, we - as a society - would obviously want these decisions to be made "fairly". But what does *fairness* even mean in this context?
- Should men and women (on average) get an identical amount of extra care?
- What if a disease has a higher prevalence in a certain group?
- What if certain groups are especially affected by a disease, but it is impossible to identify these groups based on available data?
- What if the data that are available are *biased* due to (historical or present-day) societal biases, such as poor patients having worse access to medical treatment compared to rich patients?

Once the *philosophical* question of which fairness definition to pursue is answered, various *technical* questions arise: how can the fairness of a model be quantified, and how can we actively *build* a model that is fair in the chosen sense? 

**Previously**, I worked at the [University of Lübeck](https://www.uni-luebeck.de/en/university/university.html), in the [Institute for Electrical Engineering in Medicine](https://www.ime.uni-luebeck.de/institute.html). 
In a project executed together with the research unit of [Dräger Medical](https://www.draeger.com/en-us_us/Home), we worked hard to bring *surface electromyographic monitoring of respiratory effort* into clinical practice for improving mechanical ventilation.
My research in this context spanned mathematical modeling, signal processing, parameter identification & statistical inference, all related to either surface electromyographic measurements, respiration, or both.
See my previous [publications](publications.md) for some of the work we did.


[^1]: I am not saying that I believe this is a good idea. I'm just saying: it's likely to happen, and, in fact, [already happening](https://www.science.org/doi/10.1126/science.aax2342). That does not mean that we, as a society, should't [actively decide](https://afog.berkeley.edu/programs/the-refusal-conference#overview) whether we want it to happen or not.
