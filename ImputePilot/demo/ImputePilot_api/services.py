"""
ImputePilot service layer - initial test version.
"""

class ClusteringService:
    """Clustering service."""
    
    @staticmethod
    def run_clustering(delta=0.9, rho=0.2):
        """
        Test implementation.
        Returns mock data first to validate the API wiring.
        """
        print(f"[ClusteringService] Received params: delta={delta}, rho={rho}")
        
        # Return mock clusters in the test implementation.
        clusters = [
            {'id': 1, 'name': 'Cluster 1', 'count': 100, 'rho': 0.90},
            {'id': 2, 'name': 'Cluster 2', 'count': 80, 'rho': 0.91},
            {'id': 3, 'name': 'Cluster 3', 'count': 60, 'rho': 0.89},
        ]
        
        print(f"[ClusteringService] Returning {len(clusters)} clusters")
        return {'clusters': clusters}
