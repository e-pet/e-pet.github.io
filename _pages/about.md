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

Allow me to elaborate just a tiny bit. Health risk score models are likely to be used for *resource prioritization* in the healthcare system, for example by influencing who gets access to acute care and who does not.[^1] As these are decisions that influence human livelihoods, we – as a society – would obviously want these decisions to be made "fairly". But what does *fairness* even mean in this context?
- Should men and women (on average) get an identical amount of extra care?
- What if a disease has a higher prevalence in a certain group?
- What if certain groups are especially affected by a disease, but it is impossible to identify these groups based on available data?
- What if the data that are available are *biased* due to (historical or present-day) societal biases, such as poor patients having worse access to medical treatment compared to rich patients?

Once the *philosophical* question of which fairness definition to pursue is answered, various *technical* questions arise: how can the fairness of a model be *quantified*, and how can we actively *build* a model that is fair in the chosen sense? 

**Recent news**
- I am presenting [our work on risk score fairness and metric choices](https://dl.acm.org/doi/10.1145/3593013.3594045) at the very inspiring [FAccT'23 conference](https://facctconference.org/) in Chicago! Also, we have published [a preprint on the fairness of demographic invariance in medical imaging](https://arxiv.org/abs/2305.01397) and [a PNAS commentary on what to do when complete bias removal is not an option](https://www.pnas.org/doi/10.1073/pnas.2304710120) (May/June '23).
- I got invited to participate as a speaker in a wonderful [Masterclass at the Lorentz center in Leiden on the clinical implementation of surface EMG measurements of the respiratory muscles](https://www.lorentzcenter.nl/surface-emg-of-respiratory-muscles-innovative-analyses-to-daily-practice.html) (April '23).
- I went to [MICCAI in Singapore](https://conferences.miccai.org/2022/en/) to present [our work on the impact of dataset group representation on MRI-based AD prediction performance](https://arxiv.org/abs/2204.01737) (Sep), co-organized two events on fairness and responsibility in medical ML ([1](https://faimi-workshop.github.io/), [2](https://responsibleml4healthcare.github.io/), in Oct), and presented in a wonderful session on Biases in ML at the inaugural [Danish Data Science conference](https://ddsa.dk/danishdatascience2022/) (Nov '22). 
- Two new journal papers with my previous group from Lübeck are out: [Blind source separation of inspiration and expiration in respiratory sEMG signals](https://iopscience.iop.org/article/10.1088/1361-6579/ac799c/meta) and [Model-based Estimation of Inspiratory Effort using Surface EMG](https://ieeexplore.ieee.org/abstract/document/9814853/). Also, I am now officially affiliated with the [Pioneer centre for AI](https://www.aicentre.dk/). (Jul '22)
- Our [survey paper on responsible and regulatory ML for medicine](https://doi.org/10.1109/ACCESS.2022.3178382) got accepted and published, our [MICCAI paper on feature robustness and sex differences in brain MRI](https://arxiv.org/abs/2204.01737) got accepted, I wrote about [climate change and AI](https://e-pet.github.io/posts/2022/2022-05-20-is-ai-good-for-the-planet/), and we [won funding by the DDSA](https://www.linkedin.com/feed/update/urn:li:activity:6933397159853621249/) for organizing a workshop on responsible ML for healthcare in autumn. (May/June '22)
- I had the honor of co-organizing (together with the amazing [Laura Alessandretti](https://laura.alessandretti.com/home)) a workshop on Ethical, Secure, and Just AI at the opening event for the new [Pioneer Centre for AI](https://www.aicentre.dk/). We had [an amazing list of speakers and panelists](https://twitter.com/lau_retti/status/1508417371085709313/photo/1)! (Mar '22)
- Our paper about surface EMG-based quantification of respiratory effort [got published](https://ccforum.biomedcentral.com/articles/10.1186/s13054-021-03833-w) in Critical Care (Dec '21)
- Co-organizing a [recurring seminar series on responsible AI](http://fairmed.compute.dtu.dk/seminar.php) now! Accessible via Zoom, everyone welcome. :-) (Dec '21)
- Started as a postdoc at DTU Compute with Aasa Feragen and Melanie Ganz (Sep '21)

**Previously**, I worked at the [University of Lübeck](https://www.uni-luebeck.de/en/university/university.html), in the [Institute for Electrical Engineering in Medicine](https://www.ime.uni-luebeck.de/institute.html). 
In a project executed together with the research unit of [Dräger Medical](https://www.draeger.com/en-us_us/Home), we worked hard to bring *surface electromyographic monitoring of respiratory effort* into clinical practice for improving mechanical ventilation.
My research in this context spanned mathematical modeling, signal processing, parameter identification & statistical inference, all related to either surface electromyographic measurements, respiration, or both.
See my previous [publications](publications.md) for some of the work we did.

[^1]: I am not saying that I believe this is a good idea. I'm just saying: it's likely to happen, and, in fact, [already happening](https://www.science.org/doi/10.1126/science.aax2342). That does not mean that we, as a society, should not [actively](https://afog.berkeley.edu/programs/the-refusal-conference#overview) [decide](https://www.radicalai.org/) whether we want it to happen or [not](https://bristoluniversitypress.co.uk/resisting-ai).
