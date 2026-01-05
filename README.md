# gym-agro-carbon  
## A Gym-like Environment for Optimizing Carbon Capture in Agro-Sylvo-Pastoral Systems with Batch Sequential Decision-Making

**gym-agro-carbon** is a lightweight, Gym-inspired environment designed to model **seasonal decision-making** over a **batch of agricultural parcels**. It focuses on the trade-off between **environmental sustainability** (carbon storage) and **socio-economic benefits** (agricultural yield).

This **V1** release provides a **minimal and interpretable experimental setup**, intended as a methodological baseline before coupling the learning framework with a realistic socio-ecological simulator (e.g. MAELIA implemented in GAMA).

## Objectives

This V1 environment deliberately simplifies biological and ecological processes in order to focus on the learning algorithms themselves. It serves three main purposes:

* **Algorithm Validation:**  Because V1 follows simple and explicit mathematical rules, it is possible to define an exact *oracle* strategy. This enables a precise evaluation of learning algorithms by comparing their performance against a known theoretical optimum.

* **Batch Decision-Making:**  The environment is designed to test the agent’s ability to manage an entire territory simultaneously. Rather than treating parcels independently, the agent must make **simultaneous decisions for all grid cells** ($H \times W$) at each season.

* **Baseline:**  V1 provides a controlled experimental setting to validate learning logic and implementation details before interfacing with a complex socio-ecological simulator such as GAMA (V2).

## Modeling Assumptions (V1)

To ensure algorithmic tractability and interpretability, the V1 environment relies on a set of explicit simplifications regarding space, time, and biological processes.

### 1. Spatial Structure and Context

* **Grid Representation:**  The territory is represented as a fixed-size $H \times W$ grid, where each cell corresponds to an independent agricultural plot.

* **Independence:**  In this version, plots do not interact spatially with each other.

* **Soil Context:**  Each plot is assigned a fixed soil type  $s \in \{1, \dots, 8\}.$ The soil type is **observable** by the agent and **non-controllable**, remaining constant over time.

---

### 2. Action Space

At each season, the agent selects one of four discrete actions for each plot:

1. **Fallow:** Land is left to rest without intervention.  
2. **Fallow with Manuring:** Land is left to rest with livestock integration.  
3. **Tree Planting:** An agroforestry tree is planted and initialized at age 1.  
4. **Baseline:** Conventional mono-cropping practice.

---

### 3. Temporal Dynamics

* **Discrete Time:**  One time step corresponds to a single agricultural season.
* **Tree Planting Dynamics:**  When the agent selects the *Tree Planting* action on a plot without an agent-planted tree, a new tree is introduced and its age is initialized at 1.
* **Tree Age Evolution:**  The age variable is tracked **only for trees planted by the agent**.  At each subsequent season, the age of these trees increases by one unit.
* **Pre-existing Trees (V1 Assumption):**  Some plots may contain trees at initialization.  The age of these pre-existing trees is **unknown and not explicitly modeled** in V1. Their contribution is implicitly absorbed into the baseline productivity of the plot.
* **V1 Simplifications:**  Tree mortality, competition effects, and interactions between trees are ignored in this version.

---

### 4. Reward Signal

The agent seeks to maximize a reward that balances environmental sustainability and economic viability.

#### Local Reward (per plot)

For a given plot $p$ at season $t$, the reward $r_t^{(p)}$ is defined as : $$r_t^{(p)} = \alpha \cdot C_t^{(p)} + (1 - \alpha) \cdot Y_t^{(p)}$$ 

Where :
* $C_t^{(p)}$: Carbon sequestered (environmental benefit). 
* $Y_t^{(p)}$: Agricultural yield, used here as a proxy for the socio-economic benefit. 
* $\alpha \in [0, 1]$: A public policy parameter defining the priority given to carbon capture.

#### Global Reward (territory level)

Since the agent optimizes decisions at the territorial scale, the global reward at season $t$ is defined as the sum of local rewards:

$$R_t = \sum_{p=1}^{H \times W} r_t^{(p)}.$$

## Learning Objective

The learning agent aims to optimize decision-making over a finite horizon of agricultural seasons by selecting appropriate practices for each plot in the territory.

At each season, the agent observes the current context of each plot (soil type, presence of agent-planted trees, and their age when applicable) and selects one action per plot from the available action set. Decisions are made **in batch**, covering the entire grid simultaneously.

The primary objective is to **minimize cumulative regret** with respect to an oracle strategy.

## Algorithms
This environment is initially tuned to benchmark the following algorithms in a batch setting:

- **UCB (Upper Confidence Bound):** Balances exploration and exploitation using confidence intervals.
- **Thompson Sampling:** A probabilistic approach that samples from the posterior distribution of rewards.

## Installation

This project uses **Poetry** for dependency management and requires **Python 3.12**.

### Prerequisites
- Python 3.12
- Poetry

### Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/cakane95/gym-agro-carbon.git
cd gym-agro-carbon
poetry install
```