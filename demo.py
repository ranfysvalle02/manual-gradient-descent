import numpy as np
import time

class NervousSystem:
    def __init__(self, initial_weight=0.0):
        # The 'weight' represents a deep synaptic prior.
        # e.g., The association between "Silence" and "Danger".
        self.prior_weight = initial_weight

    def forward_pass(self, context_input):
        """
        Step 1: Catch the Prediction (The Automatic Thought)
        The model hallucinates a prediction based on stale weights.
        """
        predicted_threat = self.prior_weight * context_input
        return predicted_threat

    def compute_loss(self, predicted_threat, ground_truth):
        """
        Step 2a: Calculate the felt discrepancy.
        Using Mean Squared Error: 1/2 * (Prediction - Reality)^2
        """
        return 0.5 * (predicted_threat - ground_truth)**2

    def compute_gradient(self, context_input, predicted_threat, ground_truth):
        """
        Step 2b: Calculate the Gradient.
        The vector that tells the model exactly which way the weight 
        needs to move to make the discrepancy worse (so we can subtract it).
        """
        return (predicted_threat - ground_truth) * context_input

    def cbt_session(self, context_input, ground_truth, learning_rate, session_num):
        """
        Step 3: Cognitive Restructuring (The Weight Update)
        Manually forcing the error signal to update the weights.
        """
        print(f"\n--- CBT Session {session_num} ---")
        
        # 1. Catch the Automatic Thought
        pred = self.forward_pass(context_input)
        print(f"[Step 1] Forward Pass: Context is {context_input:.1f}. Predicted threat level: {pred:.2f}")
        
        # 2. Evaluate the Evidence
        loss = self.compute_loss(pred, ground_truth)
        grad = self.compute_gradient(context_input, pred, ground_truth)
        print(f"[Step 2] Evaluate Evidence: Ground truth danger is {ground_truth:.1f}.")
        print(f"         Loss (Anxiety/Felt Error): {loss:.4f}")
        print(f"         Gradient (Direction to update): {grad:.4f}")
        
        # 3. Apply the Gradient Update
        self.prior_weight = self.prior_weight - (learning_rate * grad)
        print(f"[Step 3] Restructuring: Applying update with α={learning_rate}.")
        print(f"         New Prior Weight: {self.prior_weight:.4f}")
        time.sleep(0.5) # Simulating the metabolic cost of plasticity


def run_simulation():
    print("Initializing Biological Prediction Engine...\n")
    brain = NervousSystem()

    # =====================================================================
    # PHASE 1: Hostile Training Data (Childhood)
    # =====================================================================
    print(">>> PHASE 1: Hostile Environment (Training the Stale Prior)")
    print("In this environment, Silence (1.0) always means Danger (1.0).")
    epochs_childhood = 4
    for i in range(epochs_childhood):
        brain.cbt_session(context_input=1.0, ground_truth=1.0, learning_rate=0.5, session_num=i+1)
    
    print(f"\n[!] Model converged. Stale prior weight crystallized at: {brain.prior_weight:.2f}")
    print("[!] The model now confidently equates silence with danger to minimize loss.\n")
    time.sleep(1)

    # =====================================================================
    # PHASE 2: Adulthood (Out of Distribution)
    # =====================================================================
    print(">>> PHASE 2: Out of Distribution (The Quiet Manager)")
    manager_quiet = 1.0  # The sensory input is the same
    actual_danger = 0.0  # But the ground-truth context has changed
    
    threat = brain.forward_pass(manager_quiet)
    print(f"Manager goes quiet. Automatic Threat Prediction: {threat:.2f}")
    print("Result: Catastrophizing. The brain is executing flawlessly on stale priors.\n")
    time.sleep(1)

    # =====================================================================
    # PHASE 3: Low-Effort Therapy (Zero Learning Rate)
    # =====================================================================
    print(">>> PHASE 3: Venting Therapy (α = 0.01)")
    print("Neuromodulators are quiet. The patient is comfortable. Insight is gained, but weights don't move.")
    brain.cbt_session(context_input=manager_quiet, ground_truth=actual_danger, learning_rate=0.01, session_num=1)
    print("Result: The semantic model updated, but the amygdala didn't get the memo.\n")
    time.sleep(1)

    # =====================================================================
    # PHASE 4: Disciplined CBT & Exposure (High Learning Rate)
    # =====================================================================
    print(">>> PHASE 4: Manual Gradient Descent (α = 0.3)")
    print("Norepinephrine and Acetylcholine released. Plasticity window is artificially opened.")
    for i in range(4):
        brain.cbt_session(context_input=manager_quiet, ground_truth=actual_danger, learning_rate=0.3, session_num=i+1)

    # =====================================================================
    # PHASE 5: Regularization (Generalizing the Weights)
    # =====================================================================
    print("\n>>> PHASE 5: Regularization (Testing in the Wild)")
    print("Testing the updated model in a new, noisy environment (e.g., family dinner).")
    family_dinner_quiet = 0.8
    threat = brain.forward_pass(family_dinner_quiet)
    print(f"Input Context: {family_dinner_quiet}")
    print(f"Generalized Threat Prediction: {threat:.4f}")
    
    if threat < 0.2:
        print("Result: Success. The new weights survived a real test. The model has generalized.")
    else:
        print("Result: Overfit. The model catastrophically forgot the new weights under load.")

if __name__ == '__main__':
    run_simulation()
