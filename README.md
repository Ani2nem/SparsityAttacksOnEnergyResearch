# SparsityAttacksOnEnergyResearch
Work related to Sparsity Attacks on Energy, Dr. Yang Research


# Week 1

# Dynamic-k MoE Simulation

## Purpose
This code provides a simplified simulation of a **dynamic-k Mixture of Experts (MoE)** layer, inspired by the concepts from the *"Exploiting Activation Sparsity with Dense to Dynamic-k Mixture-of-Experts Conversion"* paper (D2DMoE).

The goal is to **understand and visualize** how a router can dynamically decide how many experts to activate per token based on a threshold parameter τ (tau), rather than using a fixed k.

## Why We Wrote This
Before diving into complex codebases like DynMoE, we wanted to:
1. **Ground the theory** from the paper in working code.
2. **Visualize** how the threshold τ affects the number of activated experts.
3. **Build intuition** about dynamic routing before analyzing real models.
4. **Have something concrete** to show progress while we continue reading and planning.

## What the Code Does

### Core Components

#### 1. Router
A small two-layer MLP that takes a token embedding and outputs a score for each expert. In D2DMoE, the router is trained via regression to predict the L2-norm of each expert's output. Here we simplify by using softmax probabilities.

#### 2. Experts
A collection of linear layers representing the "expert" networks. In a real MoE, these would be slices of the original FFN weights. Here they're simple linear layers for demonstration.

#### 3. Dynamic-k Selection Rule
Instead of always activating the top-k experts, we use the rule from D2DMoE:



