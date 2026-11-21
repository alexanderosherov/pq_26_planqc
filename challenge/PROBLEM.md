# planqc | PushQuantum Hackathon 2026

![PushQuantum Hackathon 2026, 20–22 November, Garching](planqc%20_%20PushQuantum%202026%20on%20Aqora.io%20_%20overview_files/782d55ca-4ae6-4a87-b3cf-e3bd619ecfe2_g4I1.jpg)

## Overview

Participants will implement a movement / routing algorithm for a simple two-zoned neutral atom computing
device based on the infrastructure present in:

- [Logical quantum processor based on reconfigurable atom arrays](https://www.nature.com/articles/s41586-023-06927-3) (Nature)

In the introduced architecture, only certain operations can be run in each zone. Given an input quantum
circuit, the task is to map it to the native gate set of the provided architecture, and then provide a
routing / placement algorithm that moves atoms between the two zones so that every operation runs in the
correct region. The goal is to minimize the total movement and the effective total run time of the output
circuit.

Recent techniques and algorithms are surveyed in:

- *Reuse-Aware Compilation for Zoned Quantum Architectures Based on Neutral Atoms* (IEEE)
- *ZAP: Zoned Architecture and Performant Compiler for Field-Programmable Atom Array* (IEEE)

## Purpose

**Why compilation matters.** On a neutral atom machine, the abstract circuit you write is not the circuit
that runs. The only native gates are global $R_x$, targeted $R_z$, and $CZ$ on atoms that sit next to each
other in the entangling zone. Every abstract gate therefore has to be transpiled onto this gate set, and
qubits have to be moved between zones according to the interactions they need. Moving atoms is usually the
slowest operation on the device and a significant source of errors. Two circuits that are equivalent on
paper can therefore differ a lot in run time and fidelity, depending on how well their interaction pattern
is scheduled and routed. This is what you optimize in this challenge. To make it concrete, the challenge
uses a small variational algorithm for a logistics problem as its example.

**The toy problem: choosing bus stops.** A city wants to consolidate stops along a bus route. Every
neighborhood has exactly two acceptable stops, and keeping stop $i$ incurs maintenance costs $c_i$. Find
the cheapest set of stops such that every neighborhood keeps at least one of its two.

Drawing an edge between the two acceptable stops of each neighborhood turns this into a graph problem
(a **minimum-weight vertex cover**).

The chosen instance has 12 stops and 18 neighborhoods. The graph consists of four triangles,
$\{1,2,3\}$, $\{4,5,6\}$, $\{7,8,9\}$, $\{10,11,12\}$, connected by six links:
1–4, 2–7, 3–10, 5–8, 6–11, 9–12.

The unique optimum keeps stops $\{1,2,5,6,7,8,10,12\}$ at cost 14. It drops one stop per triangle
(3, 4, 9, 11), and none of the dropped stops are linked to each other. The instance is deliberately small
enough to check by brute force: the challenge is not to solve it classically, but to run a variational
circuit for it efficiently on the device.

![Fig 0: the example instance](planqc%20_%20PushQuantum%202026%20on%20Aqora.io%20_%20overview_files/hackathon_solution_g4I1.png)

*Fig 0: The example instance. Every edge is one neighborhood and joins its two acceptable stops. The table
lists the maintenance cost $c_i$ of each stop. Green vertices and highlighted costs mark the stops kept in
the optimal solution, and every edge keeps at least one green endpoint.*

**From stops to an Ising model.** A variational algorithm needs the problem as a Hamiltonian on qubits,
one qubit per stop. Let $x_i=1$ if stop $i$ is kept, and penalize every neighborhood that loses both of
its stops:

$$F(x)=\sum_i c_i x_i+P\sum_{(i,j)\in E}(1-x_i)(1-x_j),\qquad P=5>\max_i c_i.$$

The minimum of $F$ is the cheapest valid set of stops. Substituting $x_i=(1-Z_i)/2$ (so $Z_i=-1$ for a
kept stop) gives $F=(H+146)/4$ with the classical Ising Hamiltonian

$$H=5\sum_{(i,j)\in E}Z_iZ_j+\sum_i(15-2c_i)\,Z_i,$$

which has a $ZZ$ coupling on every neighborhood and a local field on every stop.

**Where compilation comes in.** A variational circuit searches for this minimum. A generic choice is the
**hardware-efficient ansatz (HEA)**: layers of parametrized single-qubit rotations followed by $CX$ gates
between neighboring qubits, without using the problem structure. One layer on $n$ qubits needs only $n-1$
entangling gates, in two rounds of disjoint pairs. On this device, every entangling gate means that two
atoms have to meet in the entangling zone for a $CZ$ pulse, while targeted $R_z$ rotations are only
available in the storage zone. The cost is therefore decided by the number of entangling rounds and the
distance the atoms travel between them. Pairs that interact in the same round can be moved and pulsed
together.

A circuit that follows the problem structure more closely can be better on the abstract level without
being better on the device. Problem-tailored ansätze are often built on a **Max-Cut** form of the problem.
The local fields become couplings to one extra reference qubit $a$ (with weights $15-2c_i$), so that all
30 terms (18 neighborhood edges of weight 5 and 12 reference edges) are $ZZ$ interactions. A bitstring $b$
then cuts the graph with weight $W(b)=180-2F(x)$, where $x_i=b_i\oplus b_a$. The maximum cut
($W^*=152$, reached by `1100111101010` and its complement) encodes the same stops.

One example is the imaginary Hamiltonian variational ansatz (iHVA, Wang et al., *Phys. Rev. A* 111,
032612 (2025)), which places one parametrized two-qubit gate on every Max-Cut edge. Here that is 30 gates
(60 $CZ$) per layer. Because the reference qubit has 12 partners and an atom can only partner with one
neighbor per pulse, iHVA also needs at least 12 rounds per layer, compared with 2 for an HEA layer. The
best circuit in the abstract is therefore not necessarily the best one after compilation, and the metrics
below measure the latter.

**Bonus challenge: tailor the ansatz.** The cost function is fixed. The circuit that searches for its
minimum, and the way you encode the problem into qubits, are yours to choose. You are encouraged to design
an ansatz that is more expressive than a generic HEA and easier to compile than the obvious
one-gate-per-edge circuit. For example:

- **Match the hardware.** Build entangling layers from interactions the device performs in parallel,
  instead of one gate per problem edge.
- **Use the structure of the instance.** The graph is made of four tightly connected clusters. Entangling
  within clusters first keeps atoms close together and movements short.
- **Trade expressivity for cost.** Global $R_x$ and targeted $R_z$ layers need no movement of their own
  while the atoms are in storage, so extra parameters there are cheap compared with an additional round of
  entangling gates.

Please document the encoding and ansatz you choose (see [Deliverables](#deliverables)).

## PushQuantum Hackathon Challenge

Since Bluvstein et al. introduced the zoned architecture for neutral atom quantum computers, interest has
grown in how best to exploit its capabilities while respecting its constraints. How to optimize the
compilation process is still an open question. This challenge provides a neutral atom device with
simplified constraints and a series of circuits to compile.

## Quantum Device Specifications

The setup is a 100-qubit neutral atom quantum machine with a zoned architecture, as illustrated in
Figure 1.

![Figure 1: Zoned Neutral Atom Architecture](planqc%20_%20PushQuantum%202026%20on%20Aqora.io%20_%20overview_files/zone_architecture_g4I1.png)

*Figure 1: Zoned Neutral Atom Architecture*

The hackathon uses two main zones: the blue **storage zone** and the red **entangling zone**. The diagram
also shows which native gates are available:

- **Targeted $R_z$ rotations** (green circle) act on a single qubit in the storage zone.
- **Global $R_x$ rotations** (blue circles) act on all qubits in both zones, so every qubit in Fig. 1
  experiences the $R_x$ gate. *[Relaxed to also allow single-qubit targeting.]*
- **Two-qubit $CZ$ gates** (orange ovals) act transversally on all neighboring qubit pairs in the
  entangling zone. Single atoms in the entangling zone are excited by this process, but logically they only
  experience an identity operation.
- **Movement operations** are not depicted here; they are described in the next section.

You may assume that the $R_z$ and $R_x$ gates can be implemented with arbitrary angles.

## Move Semantics

![Figure 2: Movement Examples](planqc%20_%20PushQuantum%202026%20on%20Aqora.io%20_%20overview_files/slm_movement_g4I1.png)

*Figure 2: Movement Examples*

Each circle in Fig. 1, empty or filled, represents a spatial light modulator (SLM) trap that holds an atom
statically in place. Moving atoms together, apart, or between zones requires a pair of orthogonal
acousto-optic deflectors (AODs), one vertical and one horizontal. The AODs can pick up and deposit rows or
columns of atoms from and to SLM traps. Fig. 2 shows one example movement, which works as follows:

1. A contiguous set of source rows and columns is chosen. In this example, rows (5, 6) and columns (0, 1)
   are chosen.
2. The orthogonal AOD pair picks up the atoms from the SLM traps in the intersection. This intersection is
   called the **active region** of the AOD pair, and each AOD pair has a single contiguous active region.
   It is highlighted in darker green.
3. A set of empty destination rows and columns is chosen. In this example, rows (10, 11) and columns
   (0, 2) are chosen. Depositing an atom into an SLM trap that is already filled makes the move operation
   invalid.
4. Ideally, valid movements should also preserve the proportional spacing between atoms, but this is
   **not enforced** in this challenge.
5. The orthogonal AOD pair shifts to the chosen destination and deposits the carried atoms.

This movement is encoded as $[(5,6), (0,1)] \rightarrow [(10, 11), (0, 2)]$.

## Movement Constraints

You may use multiple AOD pairs simultaneously to perform parallel movements, subject to three constraints:

1. **No permutations.** Rows and columns cannot be permuted during a movement. For example, on this
   architecture the movement $[(5,6), (0,1)] \rightarrow [(10, 11), \mathbf{(2, 0)}]$ is invalid, because
   the atoms in the initial columns $(0, 1)$ are permuted. The order of rows and columns must be preserved.
2. **No ghost regions.** During pickup, movement or deposit, every intersection of simultaneous AOD pairs
   that is not an active region must be empty of atoms. These intersections are called **ghost regions**
   and are marked with red Xs in Fig. 3. The atoms marked in red would also be picked up and probably lost
   during the movement.
3. **No overlap.** While multiple AOD pairs are moving, the active region of one AOD pair cannot overlap
   with any other AOD. Fig. 4 shows an illegal move of this kind.

Consecutive move operations that violate none of these constraints can be run simultaneously. For
simplicity, movement operations are assumed to be **fully atomic**: you cannot interleave pickup, move and
deposit steps to make simultaneous movements valid. A new movement operation also cannot start until the
current set of simultaneous movements has finished.

Moves that cannot run simultaneously are run sequentially. A single movement that does not permute row or
column indices cannot violate the remaining constraints.

![Figure 3: Example of invalid simultaneous pickup operations](planqc%20_%20PushQuantum%202026%20on%20Aqora.io%20_%20overview_files/illegal_pickup_g4I1.png)

*Figure 3: Example of invalid simultaneous pickup operations.*

![Figure 4: Example of invalid simultaneous movement operations](planqc%20_%20PushQuantum%202026%20on%20Aqora.io%20_%20overview_files/illegal_move_g4I1.png)

*Figure 4: Example of invalid simultaneous movement operations.*

## Initial Atom Placement

For each transpiled circuit, atoms can initially be placed anywhere within the storage zone. For
simplicity, atom loading is assumed to happen only during initial placement. Each compiled circuit may use
any number of atoms, so using extra ancillas, or fewer atoms to simplify transpilation, is allowed. You may
also ignore atom loss during operations.

## Deliverables

### What you receive

- `move.py`: a custom Qiskit `QuantumInstruction` that defines the Move operation described in this
  document.
- `architectures/default.json`: a spec file that encodes the zones and the coordinates of the SLM traps
  per Fig. 1, together with the matching parsing method in `architecture.py`.
- `template.py`: a template file with the signature of the compiler function you must implement.
- A set of circuits for the Hardware-Efficient Ansatz implementation.

### What we want

1. A function (or series of functions) that produces a Qiskit `QuantumCircuit` object that can be used to
   solve the optimization problem. Please clearly document the encoding used.
2. An implementation of `compile_circuit` that accepts the above `QuantumCircuit` object and the parsed
   spec file, and produces an initial atom placement and a transpiled `QuantumCircuit`.

The final transpiled circuit must:

- consist only of the gate set described in [Quantum Device Specifications](#quantum-device-specifications);
- consist only of valid move operations;
- be equivalent to the input circuit up to a phase shift.

### Metrics

Compilations are ranked by these metrics, in order of importance:

1. Adherence to the strict constraints in [Movement Constraints](#movement-constraints).
2. Total Cartesian distance of moved atoms, taking parallelism into account. For each set of movement
   operations that can run in parallel, only the longest distance traveled counts.
3. Circuit depth (excluding movement operations).
4. *(Bonus)* An efficient circuit ansatz to represent the Hamiltonian.

## Prizes

- 😎 Top team wins planqc merch
