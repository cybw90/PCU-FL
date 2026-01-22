
import torch
import argparse
import os
import sys
import traceback
from config import Config
from fl_model import SimpleMLP
from pcu_algo import PCUFL

def check_pcufl_methods():
    """Diagnostic function to check PCUFL class structure"""
    try:
        from pcu_algo import PCUFL
        methods = [m for m in dir(PCUFL) if not m.startswith('_')]
        print(f"Available PCUFL methods: {methods}")
        
        if 'run' not in methods:
            print("\nERROR: 'run' method not found in PCUFL class!")
            print("The run() method needs to be added to pcu_algo.py")
            return False
        return True
    except Exception as e:
        print(f"Error importing PCUFL: {e}")
        return False

def main(args):
    # First check if PCUFL has run method
    if not check_pcufl_methods():
        print("\nFix needed in pcu_algo.py:")
        print("Add the run() method to the PCUFL class (see previous messages for implementation)")
        sys.exit(1)
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    config = Config()
    config.dataset = args.dataset
    config.num_rounds = args.rounds
    config.num_clients = args.clients
    config.epsilon_per_round = args.epsilon
    
    # Handle FP8 configuration
    if hasattr(args, 'fp8') and args.fp8:
        config.fp8_mode = args.fp8
        print(f"FP8 mode set to: {args.fp8}")
    else:
        config.fp8_mode = "none"
    
    # Set quantization parameters based on FP8 mode
    if config.fp8_mode == "none":
        config.gamma = 256
        config.alpha = 256
    elif config.fp8_mode == "E4M3":
        config.gamma = 16
        config.alpha = 256
    elif config.fp8_mode == "E5M2":
        config.gamma = 32
        config.alpha = 512
    
    # Override FP8 for very strong privacy
    if config.epsilon_per_round <= 0.5 and not args.force_fp8:
        print(f"Auto-disabling FP8 for very strong privacy (ε={config.epsilon_per_round})")
        config.fp8_mode = "none"
        config.gamma = 256
        config.alpha = 256
    
    # Handle clipping parameter
    if args.clip:
        config.clip_norm = args.clip
        print(f"Setting clip_norm to: {args.clip}")
    
    # Override gamma/alpha if specified
    if args.gamma:
        config.gamma = args.gamma
    if args.alpha:
        config.alpha = args.alpha
    
    # Set critical PCU-FL parameters
    config.w_max = 1.0 / config.clients_per_round if hasattr(config, 'clients_per_round') else 0.1
    config.eta_max = 0.0
    config.W_min = 1.0  # Ensure W_min is properly set
    
    # Validate parameters
    assert config.alpha > 0, f"Invalid alpha={config.alpha}"
    assert config.gamma > 0, f"Invalid gamma={config.gamma}"
    assert config.w_max > 0, f"Invalid w_max={config.w_max}"
    assert config.W_min > 0, f"Invalid W_min={config.W_min}"
    
    print(f"\nPCU-FL Configuration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Clients: {config.num_clients} (per round: {config.clients_per_round})")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Privacy: ε={config.epsilon_per_round}/round, δ={config.delta}")
    print(f"  FP8 mode: {config.fp8_mode}")
    print(f"  Quantization: α={config.alpha}, γ={config.gamma}")
    print(f"  Clipping: S={config.clip_norm}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Policy: w_max={config.w_max:.4f}, W_min={config.W_min}, η_max={config.eta_max}")
    
    # Load data
    print("\nLoading data...")
    train_loaders = []
    test_loader = None
    input_dim = None
    num_classes = None
    
    if config.dataset == "edge_iiot":
        from data_loader import get_edge_iiot_loaders
        
        # Load client training data
        for client_id in range(config.num_clients):
            try:
                train_loader, _, in_dim, n_classes = get_edge_iiot_loaders(config, client_id)
                if train_loader is not None:
                    train_loaders.append(train_loader)
                    if input_dim is None:
                        input_dim = in_dim
                        num_classes = n_classes
            except Exception as e:
                if client_id == 0:
                    print(f"ERROR: Failed to load data for client 0: {e}")
                    raise
                # Skip clients with data loading issues
                continue
        
        # Load server test set
        print("Loading server test set...")
        _, test_loader, _, _ = get_edge_iiot_loaders(config, client_id=None)
    
    if not train_loaders:
        print("ERROR: No training data loaded!")
        sys.exit(1)
    
    print(f"Successfully loaded data for {len(train_loaders)} clients")
    print(f"Server test set size: {len(test_loader.dataset)} samples")
    
    # Diagnostic: Check test data distribution
    if test_loader:
        from metrics_logger import PCUFLMetricsLogger
        temp_logger = PCUFLMetricsLogger(config)
        temp_logger.diagnose_data_distribution(test_loader)
    
    # Initialize model
    print(f"\nInitializing model...")
    model = SimpleMLP(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
    model = model.float()
    
    # Verify all parameters are float32
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    
    print(f"Model: SimpleMLP")
    print(f"  Input: {input_dim} → Hidden: 128 → Output: {num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize PCU-FL
    print("\n" + "="*50)
    print("Starting PCU-FL Training")
    print("="*50)
    
    try:
        pcu_fl = PCUFL(config, model, train_loaders, test_loader)
        
        # Check if run method exists
        if not hasattr(pcu_fl, 'run'):
            print("ERROR: PCUFL instance doesn't have 'run' method!")
            print("Check that run() is properly indented in the PCUFL class")
            sys.exit(1)
        
        # Run training
        summary = pcu_fl.run()
        
    except AttributeError as e:
        print(f"\nERROR: {e}")
        print("\nThis usually means the run() method is not properly defined in PCUFL class.")
        print("Check pcu_algo.py and ensure run() is indented as a class method.")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during training: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print("\n" + "="*50)
    print("PCU-FL RESULTS")
    print("="*50)
    
    print(f"\nConfiguration Summary:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Privacy: ε={config.epsilon_per_round}/round (total: {summary.get('final_epsilon', 0):.2f})")
    print(f"  FP8: {config.fp8_mode}")
    print(f"  Rounds: {summary.get('total_rounds', 0)}")
    
    print(f"\nPerformance:")
    final_acc = summary.get('final_test_acc')
    best_acc = summary.get('best_accuracy', 0)
    final_auc = summary.get('final_test_auc')
    
    if final_acc is not None:
        print(f"  Final Accuracy: {final_acc:.4f}")
    print(f"  Best Accuracy: {best_acc:.4f}")
    if final_auc is not None:
        print(f"  Final AUC: {final_auc:.4f}")
    
    print(f"\nPrivacy:")
    print(f"  Cumulative ε: {summary.get('final_epsilon', 0):.2f}")
    print(f"  Wrap events: {summary.get('total_wrap_events', 0)}")
    
    wrap_bound = summary.get('wrap_ci95_upper')
    if wrap_bound:
        print(f"  Wrap probability bound: {wrap_bound:.2e}")
    
    print("="*50)
    print(f"\nLogs: logs/")
    print(f"Plots: logs/pcufl_metrics_calibrated.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCU-FL: Proof-Carrying Updates for Federated Learning")
    
    # Basic arguments
    parser.add_argument("--dataset", type=str, default="edge_iiot",
                       choices=["edge_iiot", "nbaiot", "mnist"],
                       help="Dataset to use")
    parser.add_argument("--rounds", type=int, default=100,
                       help="Number of rounds")
    parser.add_argument("--clients", type=int, default=100,
                       help="Total clients")
    parser.add_argument("--epsilon", type=float, default=1.0,
                       help="Privacy budget per round")
    
    # FP8 arguments
    parser.add_argument("--fp8", type=str, default="none",
                       choices=["none", "E4M3", "E5M2"],
                       help="FP8 mode")
    parser.add_argument("--gamma", type=int, default=None,
                       help="Override gamma")
    parser.add_argument("--alpha", type=int, default=None,
                       help="Override alpha")
    parser.add_argument("--force-fp8", action='store_true',
                       help="Force FP8")
    
    # Clipping
    parser.add_argument("--clip", type=float, default=None,
                       help="Clip threshold")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)