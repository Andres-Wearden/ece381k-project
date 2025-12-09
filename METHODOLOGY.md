# High-Level Strategy

We evaluate topology choices using a benchmarking framework that systematically tests combinations of network structures and agent models. Each experiment is run under the same conditions, varying only the topology and model so that differences in behavior can be traced directly to these factors. The framework measures multiple dimensions at once, including accuracy, robustness to failures, and communication efficiency, which allows us to study tradeoffs rather than a single metric in isolation.

## Phase 1: Framework Development

In the first phase we designed a modular framework with clear separation between agents, network topologies, simulation logic, and evaluation. We implemented nine topology builders that generate specific graph structures, along with four agent model types that share a common interface for training and message passing. On top of this we built a simulation engine that coordinates message exchange between agents, advances the system over communication rounds, and supports controlled perturbations such as node failures and message delays. This framework forms the basis for all later experiments.

## Phase 2: Systematic Evaluation

In the second phase we used the framework to run a systematic set of experiments. We evaluated all combinations of the nine topologies and four agent models, for a total of 36 runs under matched conditions. For each run we tracked multiple metrics, including final accuracy, robustness under perturbations, error depth, network density, and message counts, over 30 communication rounds. We then introduced targeted perturbations such as node removals and added delays to test how each topology model pair responds to failures and noise. This produced a consistent dataset that captures both normal operation and stressed conditions for every configuration.

## Phase 3: Analysis and Insights

In the final phase we analyzed the collected results to extract patterns and design insights. We compared performance across topologies and models to see which combinations performed best, which were most robust, and how communication cost related to accuracy and error propagation. From these comparisons we identified tradeoffs, such as when dense connectivity helps and when it only adds overhead, and we distilled these observations into practical design rules and recommendations for choosing topologies. We also visualized key results to make differences in convergence behavior, robustness, and efficiency easier to interpret and to highlight the role of topology in shaping system level performance.

## Main Assumptions:

Our study relies on several core assumptions about communication, learning, networks, evaluation, and data that make the experiments tractable and the results comparable. We assume synchronous communication rounds in which agents exchange and process messages at the same time, with agents sharing model parameters rather than raw data over weighted and delayed edges; this reflects many federated and privacy preserving settings while giving us a clean way to study how connection strength and latency shape aggregation. On the learning side, each agent trains on a distinct local subset of the data and then combines incoming parameters through averaging (with optional weighted averaging if configured), with final predictions produced by weighted majority voting across agents; this captures realistic distributed learning where participants have different views while enforcing a common model architecture so that we can compare topologies directly. The communication network is modeled as a static directed graph of fixed size, which lets us isolate the effect of structure without confounding it with changing connectivity or agent count. For evaluation we use a common train test split, a fixed number of communication rounds, and a robustness measure based on stability under perturbations, which together provide a consistent basis for comparing accuracy and resilience across many configurations. Finally, we assume graph structured and reasonably balanced datasets so that both graph based and non graph models can be evaluated on the same tasks without extreme class imbalance dominating outcomes. These assumptions simplify the problem enough to allow systematic experimentation, while still capturing key features of practical distributed learning systems.

## Step 1: Setup

The program begins by loading a dataset with graph structure, for example the Cora citation network, which contains 2708 papers, 1433 input features per paper, and 7 label classes. It then splits this dataset into a training set and a test set, using a 70/30 split so that 70 percent of the nodes are used for learning and 30 percent are held out for evaluation.

Next, the program creates 10 agent objects of a specified model type, such as logistic regression. Each agent is initialized with the same model architecture but with its own local parameters and internal state. After that, the program builds a network topology over these 10 agents. For example, if the topology is a star, it designates one agent, say agent 0, as the hub and connects all other agents to it as spokes. Internally this creates a directed graph data structure that encodes which agents can send messages to which neighbors, along with edge weights and delays.

**Why this step matters:**

This setup step fixes the data, the agents, and the communication structure so that all later behavior can be attributed to the interaction between the model type and the chosen topology. Using a consistent split and a fixed number of agents makes comparisons across different topologies and models fair.

## Step 2: Data Distribution

Once the agents and network are created, the program distributes the training data across the 10 agents. It divides the training set into non overlapping subsets of equal size. With 1895 training samples and 10 agents, each agent receives about 189 examples. These subsets are assigned by index or by a random but reproducible partitioning function.

Each agent then stores its subset locally and prepares its model for training on that subset only. At this stage, no parameters have been exchanged between agents; they only know about their own data.

The program also distributes the test data across agents in equal non-overlapping subsets. Each agent will later make predictions only on its assigned test subset, and these predictions are aggregated across all agents to compute system-level accuracy.

## Step 3: Initial Training

In the initial training phase, the program instructs each agent to perform local training on its assigned subset. For a logistic regression agent, this might mean running gradient descent on its local examples for a fixed number of epochs. Each agent updates its parameters independently, using only its local features and labels.

After this step, the system contains 10 different models of the same type. Each agent's parameters reflect the patterns found in its local data subset, so their predictions on the shared test set will generally differ.

The program then evaluates the initial state before any communication has occurred, recording the baseline accuracy.

## Step 4: Communication Rounds (30 rounds)

The core of the program is a loop that runs for a fixed number of communication rounds, typically 30. Each round consists of several sub steps that happen in order.

### 4.1 Apply Perturbations

At the beginning of each round, before communication occurs, the program applies any configured perturbations. This can include simulating node failures by disabling an agent and preventing it from sending or receiving messages, or adding extra delays to some connections. Once a node fails, it stops sending and receiving messages, and its contribution to the voting process is removed in the evaluation phase.

**Why this sub step matters:**

By injecting failures and delays before communication, the program can measure how well each topology handles disruptions. For example, in a star, failure of the hub will strongly affect the system, while in a mesh, failure of a single node might have a smaller impact. This is how robustness is quantified in practice.

### 4.2 Communicate

After perturbations are applied, every non-failed agent prepares a message containing its current model parameters. The program then sends these messages along the outgoing edges defined by the network topology. For each directed edge from agent j to agent i, the message from j is placed in i's incoming message buffer, tagged with the associated weight and delay.

If an edge has a delay of d rounds, the message is queued so that it will only become available to the recipient after d rounds. This way, the simulation can model network latency and out of date information.

This stage simulates the real communication pattern imposed by the topology. Who sends to whom, with what weight and delay, is exactly what distinguishes a star from a ring or a mesh, and therefore directly shapes how information spreads through the system.

### 4.3 Aggregate

Once messages that are due in the current round have been received, each agent aggregates them into an updated model. The default aggregation strategy uses simple averaging, where all incoming parameters (including the agent's own parameters) are given equal weight. If weighted averaging is configured, the edge weights from incoming messages are used, but the agent's own parameters are assigned a weight equal to the average of the incoming message weights (specifically, 1.0 divided by the number of valid incoming messages), and then all weights are normalized to sum to 1.0.

For an agent i with incoming messages from neighbors j, the aggregation computes a weighted average of the parameter vectors θ_j from neighbors along with the agent's own parameters θ_i. The exact formula depends on the aggregation strategy:

- **Default (AverageAggregation)**: All parameters (neighbors + own) are averaged equally
- **WeightedAverageAggregation**: Edge weights from messages are normalized, own parameters get weight 1.0/len(neighbors), then all weights are re-normalized to sum to 1.0

The agent then replaces its old parameters with the aggregated result.

Weighted averaging (when configured) is the mechanism through which agents incorporate information from their neighbors. The weights encode connection strength, so stronger edges have more influence on the updated parameters. This step is where the topology and edge weights directly affect learning, because they determine which parameter updates are mixed and in what proportions.

### 4.4 Evaluate

After aggregation, the program evaluates the current state of the system. Each non-failed agent uses its updated model to make predictions on its assigned subset of the test set. The program then combines these predictions into a system level prediction via weighted majority voting, where agents with higher network degree (more connections) have higher voting weight.

From these combined predictions, the program computes accuracy and other metrics such as precision or recall. These metrics are recorded for the current round.

The program repeats this perturb–communicate–aggregate–evaluate cycle for all 30 rounds, storing metrics at each step.

## Simple Example: Star Topology with 3 Agents

To illustrate the process in a small setting, consider a system with three agents A0, A1, and A2, all using logistic regression models. The topology is a star, with A0 as the hub and A1 and A2 as spokes. The dataset has 100 training samples and 50 test samples. The program splits the 100 training samples into three equal parts, so each agent gets about 33 samples. The test samples are also split equally, with each agent assigned about 17 test samples.

### Initial Training

Each agent trains locally on its 33 training samples:

- A0 trains on samples 0–32 and reaches accuracy 0.60 on the test set.
- A1 trains on samples 33–65 and reaches accuracy 0.55.
- A2 trains on samples 66–99 and reaches accuracy 0.58.

At this point each model has learned patterns from only a part of the data.

### Round 1: Apply Perturbations

In this example, no perturbations are configured, so all agents remain active.

### Round 1: Communication

In the star topology, messages flow through the hub. The program sends:

- A1 → A0 with edge weight 0.8
- A2 → A0 with edge weight 0.9
- A0 → A1 with edge weight 0.8
- A0 → A2 with edge weight 0.9

These weights are stored with the parameter vectors and will determine how they are combined (if weighted averaging is used).

### Round 1: Aggregation

The program now updates each agent's parameters. Assuming the default AverageAggregation strategy (equal weights):

**A0** receives parameters from A1 and A2 and combines them with its own using equal weights:

θ_{A0}^{new} = (θ_{A1} + θ_{A2} + θ_{A0}) / 3

**A1** receives parameters only from A0 and uses:

θ_{A1}^{new} = (θ_{A0} + θ_{A1}) / 2

**A2** receives parameters only from A0 and uses:

θ_{A2}^{new} = (θ_{A0} + θ_{A2}) / 2

If WeightedAverageAggregation were used instead, the calculation would be:
- For A0: Normalize incoming weights [0.8, 0.9] → [0.471, 0.529], add own weight 1.0/2 = 0.5, re-normalize → [0.236, 0.265, 0.5], then compute weighted average
- For A1: Normalize incoming weight [0.8] → [1.0], add own weight 1.0/1 = 1.0, re-normalize → [0.5, 0.5], then compute weighted average

Each of these updates is implemented as a vector weighted sum and stored back in the agent.

**Why this matters:**

The hub A0 rapidly becomes a mixture of all agents' information, because it directly aggregates from both A1 and A2. The spokes update using the hub's parameters, so they indirectly gain access to information from the entire system through A0.

### Round 1: Evaluation

The program then evaluates all three updated models. Each agent makes predictions on its assigned test subset (about 17 samples each). The program combines these predictions via weighted majority voting, where voting weights are based on each agent's network degree. Since A0 (the hub) has degree 4 (2 incoming + 2 outgoing) and A1 and A2 each have degree 2 (1 incoming + 1 outgoing), A0's predictions have twice the weight of the spokes. The program computes a system accuracy of 0.62. This value is higher than any of the individual agents' initial accuracies (0.60, 0.55, 0.58), showing an immediate benefit from sharing information.

### Rounds 2–30

The same cycle repeats. In each round, perturbations are applied first (if any), then agents communicate their latest parameters, aggregate according to the configured strategy, and then are evaluated. Over many rounds the models move toward a consensus parameter vector that reflects information from all training subsets.

By round 30, the system may reach a final accuracy of about 0.68, which is higher than any single agent's accuracy at the start.

