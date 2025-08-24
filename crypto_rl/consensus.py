"""
Enhanced Byzantine Consensus Mechanism

This module implements a robust Byzantine fault-tolerant consensus mechanism
with signature verification for threat reports, inspired by Hyperledger Fabric's
ordering service concepts.
"""

import hashlib
from collections import defaultdict
from typing import Dict, Any, Optional

# For cryptographic signatures (conceptual)
# In a real implementation, a library like `cryptography` would be used.
class DigitalSignature:
    @staticmethod
    def sign(message: str, private_key: str) -> str:
        """Signs a message with a private key."""
        signature_input = f"{message}{private_key}".encode()
        return hashlib.sha256(signature_input).hexdigest()

    @staticmethod
    def verify(message: str, signature: str, public_key: str) -> bool:
        """Verifies a signature with a public key."""
        # In this simple model, the public_key is used like a private_key for signing.
        # This is NOT secure but serves to demonstrate the verification flow.
        expected_signature = DigitalSignature.sign(message, public_key)
        return signature == expected_signature

class EnhancedByzantineConsensus:
    """
    Implements a robust Byzantine fault-tolerant consensus mechanism
    with signature verification for threat reports.
    """
    def __init__(self, node_id: str, public_keys: Dict[str, str]):
        """
        Initialize the consensus mechanism.
        
        Args:
            node_id: The ID of the current node.
            public_keys: A dictionary mapping node IDs to their public keys.
        """
        self.node_id = node_id
        self.public_keys = public_keys
        self.reports: Dict[str, Dict[str, Any]] = {}  # {report_id: {node_id: report_data}}
        # Byzantine fault tolerance for n nodes requires 2f + 1 nodes, where f is faulty nodes.
        # A common threshold is (2/3)n + 1.
        self.threshold = int((2 * len(public_keys)) / 3) + 1

    def add_report(self, node_id: str, report: Dict[str, Any]) -> bool:
        """
        Add a threat report from a node after verifying its signature.
        
        Args:
            node_id: The ID of the reporting node.
            report: The threat report, containing 'data', 'signature', and 'report_id'.
            
        Returns:
            True if the report was added successfully, False otherwise.
        """
        if node_id not in self.public_keys:
            print(f"Consensus: Discarding report from unknown node {node_id}")
            return False
            
        public_key = self.public_keys[node_id]
        report_data_str = str(report.get('data'))
        signature = report.get('signature')
        report_id = report.get('report_id')

        if not all([report_data_str, signature, report_id]):
            print(f"Consensus: Report from {node_id} is missing required fields.")
            return False

        if not DigitalSignature.verify(report_data_str, signature, public_key):
            print(f"Consensus: Invalid signature from node {node_id}. Discarding report.")
            return False
            
        if report_id not in self.reports:
            self.reports[report_id] = {}
            
        self.reports[report_id][node_id] = report['data']
        print(f"Consensus: Added valid report from {node_id} for report_id {report_id}")
        return True

    def calculate_consensus(self, report_id: str) -> Optional[Any]:
        """
        Calculate the Byzantine fault-tolerant consensus for a given report ID.
        
        Args:
            report_id: The ID of the report to reach consensus on.
            
        Returns:
            The consensus value if reached, otherwise None.
        """
        if report_id not in self.reports:
            return None
            
        votes = defaultdict(int)
        for node_id, report_data in self.reports[report_id].items():
            # Use a JSON string to make the dictionary hashable for voting
            vote = str(report_data)
            votes[vote] += 1
            
        for vote, count in votes.items():
            if count >= self.threshold:
                print(f"Consensus reached for report '{report_id}': {vote} with {count} votes.")
                # The vote is a string representation, so we might need to convert it back
                try:
                    import json
                    return json.loads(vote.replace("'", "\"")) # Handle dict string representation
                except:
                    return vote # Return as string if not json
        
        print(f"Consensus not yet reached for report '{report_id}'. Votes: {dict(votes)}")
        return None

    def get_consensus_status(self) -> Dict[str, Any]:
        """Returns the current status of the consensus process."""
        return {
            "node_id": self.node_id,
            "total_nodes": len(self.public_keys),
            "consensus_threshold": self.threshold,
            "reports": {rid: list(nodes.keys()) for rid, nodes in self.reports.items()}
        }
