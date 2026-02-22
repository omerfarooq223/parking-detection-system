import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import io
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

# CRITICAL FIX for Gradio 5.12.0 API bug - COMPREHENSIVE PATCH
try:
    from gradio_client import utils as client_utils
    
    # Save original function
    original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    
    def fixed_json_schema_to_python_type(schema, defs=None):
        """Fixed version that handles non-dict schemas"""
        # If schema is not a dict (e.g., True, False, or other values), return generic type
        if not isinstance(schema, dict):
            return "Any"
        
        # If schema is empty dict, return Any
        if not schema:
            return "Any"
            
        # Call original function for valid schemas
        try:
            return original_json_schema_to_python_type(schema, defs)
        except Exception as e:
            print(f"⚠️ Schema parsing failed, using 'Any': {e}")
            return "Any"
    
    # Replace the function
    client_utils._json_schema_to_python_type = fixed_json_schema_to_python_type
    print("✅ Applied comprehensive Gradio API fix")
    
except Exception as e:
    print(f"⚠️ Could not apply API fix: {e}")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Color Emoji', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedParkingDetectionGUI:
    """
    Professional parking detection system with advanced recommendations
    Optimized for crowded lots and video processing
    """

    def __init__(self, model_path, model_path_2=None):
        """Initialize with trained model and Gemini API"""
        self.model = YOLO(model_path)
        print(f"✅ Model 1 loaded: {model_path}")

        self.model_2 = None
        if model_path_2 and os.path.exists(model_path_2):
            self.model_2 = YOLO(model_path_2)
            print(f"✅ Model 2 loaded: {model_path_2}")
            print("🔀 Ensemble mode enabled")
        print(f"📋 Model class names: {self.model.names}")

        # Entrance position (will be user-defined or auto-detected)
        self.entrance_position = None
        # Pixel to meter conversion (assuming standard parking space is 2.5m wide)
        self.pixels_per_meter = None
        
        # HUGGING FACE API INITIALIZATION
        self.hf_client = None
        api_key = os.getenv("HF_TOKEN")
        
        if api_key:
            try:
                self.hf_client = InferenceClient(token=api_key)
                self.hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
                print(f"✅ Hugging Face API initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: Failed to initialize Hugging Face API: {e}")
                print("⚠️ Chatbot will use fallback responses")
        else:
            print("⚠️ Warning: HF_TOKEN not found")
            print("⚠️ Chatbot will use fallback responses")
            print("💡 To enable Hugging Face chatbot:")
            print("   1. Get your token from: https://huggingface.co/settings/tokens")
            print("   2. Add as a Space secret: HF_TOKEN=your_token_here")


    def ensemble_predict(self, image_path, conf_threshold=0.4, iou_threshold=0.5):
        """
        Combine predictions from both models using weighted boxes fusion
        """
        if self.model_2 is None:
            # Only one model, use normal prediction
            return self.model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]
    
        # Get predictions from both models
        results_1 = self.model.predict(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
         )[0]
    
        results_2 = self.model_2.predict(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]
    
        # Simple voting: combine all detections
        boxes_1 = results_1.boxes.xyxy.cpu().numpy()
        scores_1 = results_1.boxes.conf.cpu().numpy()
        classes_1 = results_1.boxes.cls.cpu().numpy()
    
        boxes_2 = results_2.boxes.xyxy.cpu().numpy()
        scores_2 = results_2.boxes.conf.cpu().numpy()
        classes_2 = results_2.boxes.cls.cpu().numpy()
    
        # Combine all boxes
        if len(boxes_2) > 0:
            all_boxes = np.vstack([boxes_1, boxes_2]) 
            all_scores = np.hstack([scores_1, scores_2]) 
            all_classes = np.hstack([classes_1, classes_2]) 
        else:
            all_boxes = boxes_1
            all_scores = scores_1
            all_classes = classes_1

        indices = cv2.dnn.NMSBoxes(
            all_boxes.tolist(),
            all_scores.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )

        # Filter boxes based on NMS results
        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes = all_boxes[indices]
            final_scores = all_scores[indices]
            final_classes = all_classes[indices]
        else:
            final_boxes = all_boxes
            final_scores = all_scores
            final_classes = all_classes

        results_1.boxes.xyxy = torch.tensor(final_boxes)
        results_1.boxes.conf = torch.tensor(final_scores)
        results_1.boxes.cls = torch.tensor(final_classes)

        print(f"🔀 Ensemble: Model1={len(boxes_1)}, Model2={len(boxes_2)}, Combined={len(final_boxes)} detections")
    
        return results_1

    def get_model_info(self):
        """Return comprehensive model training information"""
        return {
            'model_name': 'YOLOv8',
            'model_variant': 'YOLOv8m (Medium)',
            'framework': 'Ultralytics YOLO v8.3.248',
            'epochs_trained': 40,
            'batch_size': 16,
            'image_size': 640,
            'dataset_size': {
                'train': 4023,
                'validation': 387,
                'test': 387
            },
            'classes': ['car (occupied)', 'free (empty space)'],
            'optimizer': 'AdamW (Auto-selected)',
            'learning_rate': 0.001667,
            'momentum': 0.9,
            'augmentations': [
                'Mosaic augmentation',
                'Random horizontal flip (50%)',
                'HSV color jittering',
                'Random translation (10%)',
                'Random scaling (50%)',
                'Blur (1% probability)',
                'MedianBlur (1% probability)',
                'ToGray (1% probability)',
                'CLAHE contrast enhancement (1%)'
            ],
            'training_time': '~1 hour (61 minutes)',
            'hardware': 'Google Colab - Tesla T4 GPU (15GB VRAM)',
            'final_metrics': {
                'precision': 0.952,
                'recall': 0.938,
                'mAP50': 0.975,
                'mAP50-95': 0.765
            },
            'class_metrics': {
                'car': {
                    'precision': 0.977,
                    'recall': 0.959,
                    'mAP50': 0.987,
                    'mAP50-95': 0.788
                },
                'free': {
                    'precision': 0.927,
                    'recall': 0.919,
                    'mAP50': 0.962,
                    'mAP50-95': 0.741
                }
            },
            'loss_functions': [
                'Box loss (localization) - Final: 0.737',
                'Class loss (classification) - Final: 0.384',
                'DFL loss (distribution focal) - Final: 0.986'
            ],
            'model_size': '22.5 MB',
            'inference_speed': '~5ms per image',
            'total_parameters': '11,126,358',
            'flops': '28.4 GFLOPs'
        }

    def analyze_and_visualize(self, image, confidence_threshold=0.30,
                             show_zones=True, show_recommendations=True, 
                             entrance_point_coords=None, use_aggressive_detection=False, iou_threshold=0.5):
        """
        Main function with enhanced features and aggressive detection mode.

        Parameters:
        - entrance_point_coords: Tuple (x, y) in pixels from the Gradio click event.
        - use_aggressive_detection: If True, uses more aggressive settings for crowded/unsymmetric lots
        """

        if isinstance(image, Image.Image):
            image = np.array(image)

        img_height, img_width = image.shape[:2]

        # Set entrance position: user-defined if available, else default to bottom-center
        if entrance_point_coords is not None and len(entrance_point_coords) == 2:
            self.entrance_position = entrance_point_coords
        else:
            self.entrance_position = (img_width / 2, img_height)

        # Adjust parameters for aggressive detection mode
        if use_aggressive_detection:
            adjusted_confidence = max(0.05, confidence_threshold - 0.30)
            adjusted_iou = min(iou_threshold, 0.3)  # More aggressive NMS
            max_detections = 300
            agnostic_nms = True
            print(f"🔍 Aggressive detection enabled: conf={adjusted_confidence:.2f}, iou={iou_threshold}, max_det={max_detections}")
        else:
            adjusted_confidence = confidence_threshold
            adjusted_iou = iou_threshold
            max_detections = 200
            agnostic_nms = False

        # Save temporarily for prediction
        temp_path = 'temp_input.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Run detection with optimized post-processing
        results = self.model.predict(
            temp_path,
            conf=adjusted_confidence,
            iou=adjusted_iou,
        )
        
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # Extract and filter detections
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # Separate empty and occupied with size filtering
        empty_spaces = []
        occupied_spaces = []

        # Calculate median area for filtering outliers
        all_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in xyxy]
        if len(all_areas) > 0:
            median_area = np.median(all_areas)
            # More lenient filtering for aggressive mode
            if use_aggressive_detection:
                min_area = median_area * 0.2
                max_area = median_area * 4.0
            else:
                min_area = median_area * 0.3
                max_area = median_area * 3.0

            # Estimate pixels per meter
            median_width = np.median([box[2] - box[0] for box in xyxy])
            if median_width > 0:
                self.pixels_per_meter = median_width / 2.5
            else:
                self.pixels_per_meter = 50
        else:
            min_area, max_area = 0, float('inf')
            self.pixels_per_meter = 50

        for box, cls, conf in zip(xyxy, classes, confidences):
            area = (box[2] - box[0]) * (box[3] - box[1])

            # Size filtering
            if area < min_area or area > max_area:
                continue

            space_info = {
                'bbox': box,
                'confidence': conf,
                'center': np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]),
                'area': area,
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'id': None
            }

            # Class 1 = free (empty), Class 0 = car (occupied)
            if cls == 1:
                empty_spaces.append(space_info)
            elif cls == 0:
                occupied_spaces.append(space_info)

        # Assign spot IDs based on position
        all_spaces = empty_spaces + occupied_spaces
        self.assign_spot_ids(all_spaces, img_width, img_height)

        # Create enhanced visualization
        visualized_image = self.create_enhanced_visualization(
            image, empty_spaces, occupied_spaces,
            show_zones, show_recommendations
        )

        # Generate enhanced statistics
        stats = self.generate_enhanced_statistics(empty_spaces, occupied_spaces, use_aggressive_detection)

        # Generate smart recommendations
        recommendations = self.generate_smart_recommendations(
            empty_spaces, occupied_spaces
        )
        self.store_analysis_results(empty_spaces, occupied_spaces, stats, recommendations)

        return visualized_image, stats, recommendations

    def assign_spot_ids(self, spaces, img_width, img_height):
        """Assign grid-based IDs to parking spots"""

        if len(spaces) == 0:
            return

        # Create 3x3 grid for zones
        grid_size = 3
        zone_width = img_width / grid_size
        zone_height = img_height / grid_size

        # Group spaces by zone and assign IDs
        for space in spaces:
            cx, cy = space['center']

            # Determine zone (A1, A2, A3, B1, B2, B3, C1, C2, C3)
            col = int(cx // zone_width)
            row = int(cy // zone_height)

            # Ensure within bounds
            col = min(col, grid_size - 1)
            row = min(row, grid_size - 1)

            zone_letter = chr(65 + row)  # A, B, C
            zone_number = col + 1  # 1, 2, 3

            # Find position within zone
            zone_spaces = [s for s in spaces
                          if int(s['center'][0] // zone_width) == col
                          and int(s['center'][1] // zone_height) == row]

            # Sort by position within zone
            zone_spaces.sort(key=lambda s: (s['center'][1], s['center'][0]))
            spot_num = next(i for i, s in enumerate(zone_spaces) if s is space) + 1

            space['id'] = f"{zone_letter}{zone_number}-{spot_num}"

    def pixels_to_meters(self, pixels):
        """Convert pixels to meters"""
        if self.pixels_per_meter is None:
            return pixels / 50
        return pixels / self.pixels_per_meter

    def find_clusters(self, empty_spaces, eps=100, min_samples=2):
        """Find clusters of empty spaces using DBSCAN"""

        if len(empty_spaces) < 2:
            return []

        centers = np.array([s['center'] for s in empty_spaces])
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)

        clusters = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue

            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_spaces = [empty_spaces[i] for i in cluster_indices]
            cluster_center = np.mean([s['center'] for s in cluster_spaces], axis=0)

            clusters.append({
                'spaces': cluster_spaces,
                'center': cluster_center,
                'size': len(cluster_spaces)
            })

        return clusters

    def calculate_smart_score(self, space, empty_spaces, occupied_spaces, clusters):
        """Enhanced scoring algorithm with multiple factors"""

        score = 0
        reasons = []

        # Factor 1: Detection confidence (0-25 points)
        conf_score = space['confidence'] * 25
        score += conf_score
        if space['confidence'] > 0.85:
            reasons.append(f"High confidence ({space['confidence']:.0%})")
        elif space['confidence'] > 0.7:
            reasons.append(f"Good confidence ({space['confidence']:.0%})")
        else:
            reasons.append(f"Moderate confidence ({space['confidence']:.0%})")

        # Factor 2: Distance from entrance (0-30 points)
        if self.entrance_position is not None:
            distance_to_entrance = np.linalg.norm(
                space['center'] - np.array(self.entrance_position)
            )
            img_diagonal_estimate = np.sqrt((1000**2) + (750**2)) * 2
            normalized_distance = distance_to_entrance / img_diagonal_estimate

            entrance_score = 30 * (1 - normalized_distance)
            score += entrance_score

            if normalized_distance < 0.3:
                reasons.append("Very close to entrance")
            elif normalized_distance < 0.6:
                reasons.append("Moderate distance from entrance")

        # Factor 3: Cluster bonus (0-20 points)
        in_cluster = False
        cluster_size = 0
        for cluster in clusters:
            if any(s is space for s in cluster['spaces']):
                in_cluster = True
                cluster_size = cluster['size']
                cluster_score = min(20, cluster_size * 4)
                score += cluster_score
                reasons.append(f"In cluster of {cluster_size} spaces")
                break

        if not in_cluster and len(clusters) > 0:
            score -= 5
            reasons.append("Isolated spot")

        # Factor 4: Space size (0-15 points)
        if len(empty_spaces) > 1:
            median_area = np.median([s['area'] for s in empty_spaces])
            size_ratio = space['area'] / median_area

            if size_ratio > 1.2:
                size_score = 15
                score += size_score
                reasons.append("Larger than average")
            elif size_ratio > 1.0:
                size_score = 10
                score += size_score
                reasons.append("Good size")

        # Factor 5: Accessibility (0-10 points)
        nearby_occupied = 0
        for occ in occupied_spaces:
            dist = np.linalg.norm(space['center'] - occ['center'])
            if dist < 150:
                nearby_occupied += 1

        if nearby_occupied <= 1:
            score += 10
            reasons.append("Easy access")
        elif nearby_occupied <= 2:
            score += 5
            reasons.append("Moderate access")
        else:
            reasons.append("Tight access")

        # Factor 6: Edge detection penalty
        img_margin = 50
        x, y = space['center']
        is_edge = (x < img_margin or y < img_margin)

        if is_edge:
            score -= 5
            reasons.append("Edge location")

        return max(0, min(100, score)), reasons

    def rank_spaces_advanced(self, empty_spaces, occupied_spaces):
        """Advanced ranking with clustering and smart scoring"""

        if len(empty_spaces) == 0:
            return []

        clusters = self.find_clusters(empty_spaces)

        recommendations = []
        for space in empty_spaces:
            score, reasons = self.calculate_smart_score(
                space, empty_spaces, occupied_spaces, clusters
            )

            recommendations.append({
                'space': space,
                'score': score,
                'reasons': reasons
            })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

    def create_enhanced_visualization(self, image, empty_spaces, occupied_spaces,
                                     show_zones, show_recommendations):
        """Create professional visualization with all enhancements"""

        img = image.copy()
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(img)
        ax.axis('off')

        # Draw occupied spaces
        for space in occupied_spaces:
            box = space['bbox']
            rect = Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='#DC143C',
                facecolor='#DC143C', alpha=0.35
            )
            ax.add_patch(rect)

        # Draw empty spaces
        for space in empty_spaces:
            box = space['bbox']
            rect = Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2.5, edgecolor='#00CED1',
                facecolor='#00CED1', alpha=0.4
            )
            ax.add_patch(rect)

            # Add spot ID
            if space['id']:
                cx, cy = space['center']
                ax.text(cx, cy, space['id'], color='black', fontsize=9,
                       fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Show recommendations
        if show_recommendations and len(empty_spaces) > 0:
            recommendations = self.rank_spaces_advanced(empty_spaces, occupied_spaces)[:5]
            colors = ['#FFD700', '#32CD32', '#FFFF00', '#FFA500', '#90EE90']

            for i, rec in enumerate(recommendations):
                space = rec['space']
                box = space['bbox']

                # Enhanced border
                rect = FancyBboxPatch(
                    (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                    linewidth=5, edgecolor=colors[i], facecolor='none',
                    boxstyle="round,pad=5", zorder=10
                )
                ax.add_patch(rect)

                # Rank badge
                cx, cy = space['center']
                circle = Circle((cx, cy), 30, color=colors[i], ec='black',
                              linewidth=3, zorder=11)
                ax.add_patch(circle)

                # Rank number
                ax.text(cx, cy, str(i+1), color='black', fontsize=18,
                       fontweight='bold', ha='center', va='center', zorder=12)

                # Score
                score_text = f"{rec['score']:.0f}"
                ax.text(cx, cy + 45, score_text, color='white', fontsize=11,
                       fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8),
                       zorder=12)

                # Distance in meters
                if self.entrance_position is not None:
                    dist_pixels = np.linalg.norm(space['center'] - np.array(self.entrance_position))
                    dist_meters = self.pixels_to_meters(dist_pixels)
                    dist_text = f"~{dist_meters:.0f}m"
                    ax.text(box[0], box[1] - 10, dist_text, color='white', fontsize=10,
                           fontweight='bold', ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.9))

                # Spot ID
                if space['id']:
                    ax.text(box[2], box[1] - 10, space['id'], color='white', fontsize=10,
                           fontweight='bold', ha='right', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.9))

            # Draw walking paths to top 3
            if self.entrance_position is not None and len(recommendations) >= 1:
                entrance_point = np.array(self.entrance_position)

                for i in range(min(3, len(recommendations))):
                    space = recommendations[i]['space']
                    space_center = space['center']

                    arrow = FancyArrowPatch(
                        entrance_point, space_center,
                        arrowstyle='->',
                        linestyle='--', 
                        linewidth=2.5,
                        color=colors[i],
                        alpha=0.7,
                        zorder=9,
                        mutation_scale=20,
                        connectionstyle="arc3,rad=0.2"
                    )
                    ax.add_patch(arrow)

        # Show entrance
        if self.entrance_position is not None and show_recommendations:
            ex, ey = self.entrance_position
            entrance_circle = Circle((ex, ey), 40, color='blue', ec='white',
                                    linewidth=3, zorder=15, alpha=0.8)
            ax.add_patch(entrance_circle)
            ax.text(ex, ey, '🚪', fontsize=25, ha='center', va='center', zorder=16)
            ax.text(ex, ey - 60, 'ENTRANCE', color='white', fontsize=12,
                   fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.9),
                   zorder=16)

        # Enhanced zones
        if show_zones and (len(empty_spaces) + len(occupied_spaces)) > 0:
            self.draw_enhanced_zones(ax, img.shape, empty_spaces, occupied_spaces)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#DC143C', alpha=0.5, label=f'Occupied ({len(occupied_spaces)})'),
            Patch(facecolor='#00CED1', alpha=0.5, label=f'Empty ({len(empty_spaces)})')
        ]
        if show_recommendations and len(empty_spaces) > 0:
            legend_elements.append(Patch(facecolor='#FFD700', label='🥇 Top Pick'))

        ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                 framealpha=0.9, edgecolor='black', fancybox=True)

        # Statistics overlay
        total = len(empty_spaces) + len(occupied_spaces)
        if total > 0:
            occupancy = (len(occupied_spaces) / total) * 100
            stats_text = f"Total: {total} | Available: {len(empty_spaces)} | Occupancy: {occupancy:.0f}%"
            ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=13, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.85),
                   color='white')

        # Convert to image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()

        return result_image

    def draw_enhanced_zones(self, ax, img_shape, empty_spaces, occupied_spaces):
        """Enhanced zone visualization with detailed info"""

        height, width = img_shape[:2]
        grid_size = 3
        zone_width = width / grid_size
        zone_height = height / grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                x1, y1 = col * zone_width, row * zone_height
                x2, y2 = (col + 1) * zone_width, (row + 1) * zone_height

                empty_in_zone = sum(1 for s in empty_spaces
                                  if x1 <= s['center'][0] < x2 and y1 <= s['center'][1] < y2)
                occupied_in_zone = sum(1 for s in occupied_spaces
                                     if x1 <= s['center'][0] < x2 and y1 <= s['center'][1] < y2)
                total_in_zone = empty_in_zone + occupied_in_zone

                if total_in_zone > 0:
                    occupancy = occupied_in_zone / total_in_zone

                    if occupancy < 0.3:
                        color, label = '#00FF00', '✓ Available'
                    elif occupancy < 0.7:
                        color, label = '#FFD700', '⚠ Limited'
                    else:
                        color, label = '#FF4500', '✗ Full'

                    rect = Rectangle((x1, y1), zone_width, zone_height,
                                   linewidth=2, edgecolor='white',
                                   facecolor=color, alpha=0.15, linestyle='--')
                    ax.add_patch(rect)

                    # Zone label
                    zone_name = f"Zone {chr(65+row)}{col+1}"
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(cx, cy, f"{zone_name}\n{empty_in_zone}/{total_in_zone} free",
                           color='white', fontsize=10, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

    def generate_enhanced_statistics(self, empty_spaces, occupied_spaces, aggressive_mode=False):
        """Generate detailed statistics"""

        total = len(empty_spaces) + len(occupied_spaces)

        if total == 0:
            return "❌ No parking spaces detected.\n\n💡 Suggestions:\n• Try 'Aggressive Detection' mode\n• Lower confidence threshold\n• Check image quality\n• Ensure aerial view"

        empty_count = len(empty_spaces)
        occupied_count = len(occupied_spaces)
        occupancy_rate = (occupied_count / total) * 100
        availability_rate = (empty_count / total) * 100

        if empty_count == 0:
            status = "🔴 LOT FULL"
            recommendation = "❌ No spaces available. Try nearby lots."
        elif empty_count < total * 0.15:
            status = "🟠 VERY LIMITED"
            recommendation = "⚠️ Only a few spaces left. Act quickly!"
        elif empty_count < total * 0.4:
            status = "🟡 MODERATE"
            recommendation = "👍 Decent availability. Check recommendations."
        else:
            status = "🟢 AVAILABLE"
            recommendation = "✅ Plenty of spaces. Pick your preference!"

        if empty_count > 0:
            avg_confidence = np.mean([s['confidence'] for s in empty_spaces])
            confidence_rating = "High" if avg_confidence > 0.8 else "Good" if avg_confidence > 0.65 else "Moderate"
        else:
            avg_confidence = 0
            confidence_rating = "N/A"

        clusters = self.find_clusters(empty_spaces)
        
        mode_indicator = " (🔍 Aggressive Mode)" if aggressive_mode else ""

        stats = f"""
╔══════════════════════════════════════════╗
║     📊 PARKING LOT ANALYSIS REPORT{mode_indicator}       ║
╚══════════════════════════════════════════╝

🚦 STATUS: {status}
{recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 CAPACITY METRICS:
   Total Spaces:          {total}
   ├─ 🟢 Available:       {empty_count} spaces ({availability_rate:.1f}%)
   └─ 🔴 Occupied:        {occupied_count} spaces ({occupancy_rate:.1f}%)

   Occupancy Rate:        {occupancy_rate:.1f}%
   Availability Rate:     {availability_rate:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 DETECTION QUALITY:
   Avg Confidence:        {avg_confidence:.1%} ({confidence_rating})
   Empty Space Clusters:  {len(clusters)} found
"""

        if len(clusters) > 0:
            largest_cluster = max(clusters, key=lambda x: x['size'])
            stats += f"   Largest Cluster:       {largest_cluster['size']} spaces\n"

        stats += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        stats += """
💡 INSIGHTS:
   • Check recommendations for best spots
   • Top picks are scored based on multiple factors
   • Click on image to set entrance location
"""

        if aggressive_mode:
            stats += "   • 🔍 Aggressive detection used for complex scenes\n"

        if occupancy_rate > 80:
            stats += "   • ⚠️ High occupancy - arrive early next time\n"
        elif occupancy_rate < 30:
            stats += "   • ✅ Low occupancy - great time to visit\n"

        return stats

    def generate_smart_recommendations(self, empty_spaces, occupied_spaces):
        """Generate intelligent recommendations"""

        if len(empty_spaces) == 0:
            return """
╔══════════════════════════════════════════╗
║        🚫 NO SPACES AVAILABLE            ║
╚══════════════════════════════════════════╝

The parking lot is currently full.

💡 ALTERNATIVE OPTIONS:
   • Wait for a vehicle to leave
   • Check nearby parking facilities
   • Consider street parking
   • Use public transportation

⏰ TYPICAL BUSY TIMES:
   Peak hours usually see highest occupancy.
   Try visiting during off-peak hours.
"""

        recommendations = self.rank_spaces_advanced(empty_spaces, occupied_spaces)[:5]

        rec_text = f"""
╔══════════════════════════════════════════╗
║      🎯 SMART PARKING RECOMMENDATIONS    ║
╚══════════════════════════════════════════╝

Found {len(empty_spaces)} available spaces!
Showing top {min(5, len(recommendations))} recommendations:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

        medals = ['🥇', '🥈', '🥉', '4', '5']
        quality_labels = ['BEST CHOICE', 'EXCELLENT', 'VERY GOOD', 'GOOD', 'ALTERNATIVE']

        for i, rec in enumerate(recommendations):
            space = rec['space']
            score = rec['score']

            if score >= 80:
                quality = "⭐⭐⭐⭐⭐ EXCELLENT"
            elif score >= 65:
                quality = "⭐⭐⭐⭐ VERY GOOD"
            elif score >= 50:
                quality = "⭐⭐⭐ GOOD"
            else:
                quality = "⭐⭐ FAIR"

            rec_text += f"{medals[i]} RECOMMENDATION #{i+1} - {quality_labels[i]}\n"
            rec_text += f"{'─' * 40}\n"
            rec_text += f"   Spot ID:           {space['id']}\n"
            rec_text += f"   Overall Score:     {score:.0f}/100 {quality}\n"
            rec_text += f"   Detection Conf:    {space['confidence']:.1%}\n"

            if self.entrance_position is not None:
                dist_pixels = np.linalg.norm(space['center'] - np.array(self.entrance_position))
                dist_meters = self.pixels_to_meters(dist_pixels)
                walk_time = int(dist_meters / 1.4 * 60)
                rec_text += f"   Distance:          ~{dist_meters:.0f}m from entrance (~{walk_time}s walk)\n"

            rec_text += f"\n   ✓ Key Benefits:\n"
            for reason in rec['reasons']:
                rec_text += f"     • {reason}\n"

            rec_text += "\n"

        rec_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        rec_text += "💡 HOW TO USE:\n"
        rec_text += "   1. Click on image to set entrance location\n"
        rec_text += "   2. Look for numbered circles (1-5) on the image\n"
        rec_text += "   3. Gold border with #1 = Top recommendation\n"
        rec_text += "   4. Dashed arrows show walking paths from entrance\n"
        rec_text += "   5. Distances shown in meters with walking time\n"
        rec_text += "   6. For crowded lots, enable 'Aggressive Detection'\n"

        return rec_text

    def store_analysis_results(self, empty_spaces, occupied_spaces, stats, recommendations):
        """Store analysis results for chatbot queries"""
        self.last_empty_count = len(empty_spaces)
        self.last_occupied_count = len(occupied_spaces)
        self.last_stats = stats
        self.last_recommendations = recommendations
        self.last_total = len(empty_spaces) + len(occupied_spaces)

        self.raw_empty_spaces = empty_spaces
        self.raw_occupied_spaces = occupied_spaces

    def get_api_data(self):
        """API Endpoint: Returns clean JSON data for external applications"""
        if not hasattr(self, 'raw_empty_spaces'):
            return {"error": "No image analyzed yet"}
            
        return {
            "status": "success",
            "timestamp": "latest",
            "summary": {
                "total_spaces": self.last_total,
                "available": self.last_empty_count,
                "occupied": self.last_occupied_count,
                "occupancy_rate": round((self.last_occupied_count / self.last_total * 100), 2) if self.last_total > 0 else 0
            },
            "available_spots": [
                {
                    "id": space.get('id', 'Unknown'),
                    "confidence": space.get('confidence', 0.0),
                    "bbox": space.get('bbox') if isinstance(space.get('bbox'), list) else space.get('bbox').tolist() 
                } 
                for space in self.raw_empty_spaces
            ]
        }

    def chat_with_bot(self, message, history):
        """Handle chatbot conversations about parking using Gemini API"""
        
        # Check if analysis has been done
        if not hasattr(self, 'last_stats') or self.last_stats is None:
            return "👋 Hi! Please upload and analyze a parking lot image first using the **Image Analysis** tab, then come back here to ask me questions! 📸"
        
        # Prepare context about current parking situation
        model_info = self.get_model_info()
        context = f"""
You are a helpful AI parking assistant for a Smart Parking Detection System. Here is the current parking lot analysis:

CURRENT PARKING STATUS:
- Total parking spaces: {self.last_total}
- Available (empty) spaces: {self.last_empty_count}
- Occupied spaces: {self.last_occupied_count}
- Occupancy rate: {(self.last_occupied_count / self.last_total * 100) if self.last_total > 0 else 0:.1f}%

SYSTEM CAPABILITIES:
- AI Model: {model_info['model_name']} ({model_info['model_variant']})
- Detection Accuracy: {model_info['final_metrics']['mAP50']*100:.1f}%
- Training: {model_info['epochs_trained']} epochs on {model_info['dataset_size']['train']} images
- Classes Detected: {', '.join(model_info['classes'])}
- Dataset: PKLot dataset (custom parking lot images)
- Training Hardware: {model_info['hardware']}
- Training Duration: {model_info['training_time']}

DETECTION TECHNOLOGY:
The system uses YOLOv8 (You Only Look Once) object detection:
1. The neural network analyzes the parking lot image
2. It detects bounding boxes around cars and empty spaces
3. Confidence scores are calculated for each detection
4. Post-processing filters false detections
5. The system ranks empty spots based on distance from entrance, clustering, size, and accessibility

RECOMMENDATIONS AVAILABLE:
The system provides ranked recommendations based on:
1. Distance from entrance
2. Detection confidence
3. Space clustering (nearby empty spots)
4. Space size
5. Accessibility (easy entry/exit)

INSTRUCTIONS FOR RESPONSES:
- Be friendly, helpful, and conversational
- Use emojis appropriately to make responses engaging
- Keep responses concise but informative (2-4 paragraphs max)
- If asked about availability, mention the actual numbers
- If asked about recommendations, explain the top spot briefly
- If asked about technical details, provide accurate info from the model_info above
- If asked how the system works, explain the YOLO detection and scoring algorithm
- If asked about the dataset, mention it's based on PKLot dataset with custom parking lot images
- If asked about training details (epochs, hardware, etc), use the exact information provided above
- Always be positive and helpful in tone

The user is asking about parking availability, recommendations, or system information.
"""
        # Use Hugging Face API if available
        if self.hf_client:
            try:
                # Format conversation history for Hugging Face
                messages = []
        
                # Add system context
                messages.append({
                    "role": "system",
                    "content": context
                })
        
                # Add conversation history
                if isinstance(history, list) and len(history) > 0:
                    for item in history:
                        if isinstance(item, tuple) and len(item) == 2:
                            messages.append({"role": "user", "content": item[0]})
                            messages.append({"role": "assistant", "content": item[1]})
                        elif isinstance(item, dict):
                            messages.append(item)
        
                # Add current user message
                messages.append({"role": "user", "content": message})
        
                # Get response from Hugging Face
                response = self.hf_client.chat_completion(
                    messages=messages,
                    model=self.hf_model,
                    max_tokens=500,
                    temperature=0.7
                )
        
                return response.choices[0].message.content
        
            except Exception as e:
                print(f"⚠️ Hugging Face API error: {e}")
                print("⚠️ Falling back to rule-based responses")

        # Enhanced fallback responses
        message_lower = message.lower()
        
        # Training-related questions
        if any(word in message_lower for word in ['train', 'epoch', 'how many epoch', 'training time', 'trained']):
            return f"""🤖 **Model Training Details:**

**Training Configuration:**
• Architecture: {model_info['model_name']} ({model_info['model_variant']})
• Epochs: {model_info['epochs_trained']} epochs
• Batch Size: {model_info['batch_size']}
• Training Images: {model_info['dataset_size']['train']} images
• Validation Images: {model_info['dataset_size']['validation']} images

**Hardware & Duration:**
• Platform: {model_info['hardware']}
• Training Time: {model_info['training_time']}
• Optimizer: {model_info['optimizer']}

**Results:**
• Final mAP50: {model_info['final_metrics']['mAP50']*100:.1f}%
• Precision: {model_info['final_metrics']['precision']*100:.1f}%
• Recall: {model_info['final_metrics']['recall']*100:.1f}%"""
        
        # Dataset questions
        elif any(word in message_lower for word in ['dataset', 'data', 'images', 'how train']):
            return f"""📊 **Training Dataset Information:**

**Dataset Composition:**
• Training Set: {model_info['dataset_size']['train']} images
• Validation Set: {model_info['dataset_size']['validation']} images
• Test Set: {model_info['dataset_size']['test']} images

**Dataset Source:**
Based on PKLot dataset with custom parking lot images from various angles and lighting conditions.

**Classes:**
• {model_info['classes'][0]} (occupied parking spaces)
• {model_info['classes'][1]} (empty parking spaces)

**Data Augmentation Applied:**
{chr(10).join('• ' + aug for aug in model_info['augmentations'][:5])}

This diverse dataset helps the model work in different parking lot conditions!"""
        
        # Detection/How it works questions
        elif any(word in message_lower for word in ['detect', 'detection', 'how does', 'how work', 'technology', 'algorithm']):
            return f"""🔍 **How the Detection System Works:**

**Step 1: Image Analysis**
The YOLOv8 neural network processes the parking lot image in a single pass, detecting both cars and empty spaces simultaneously.

**Step 2: Object Detection**
• The model identifies bounding boxes around each parking space
• Confidence scores are calculated (current avg: {np.mean([s['confidence'] for s in self.raw_empty_spaces]) if hasattr(self, 'raw_empty_spaces') and self.raw_empty_spaces else 0:.1%})
• Non-maximum suppression removes duplicate detections

**Step 3: Smart Ranking**
Empty spaces are scored based on:
• Distance from entrance (closer = better)
• Space clustering (groups of empty spots)
• Accessibility (easy to maneuver)
• Detection confidence
• Space size

**Performance:**
• Inference Speed: {model_info['inference_speed']}
• Accuracy: {model_info['final_metrics']['mAP50']*100:.1f}% mAP50"""
        
        # Available/Empty spaces
        elif any(word in message_lower for word in ['available', 'empty', 'free', 'space', 'spot', 'how many']):
            if self.last_empty_count == 0:
                return "🔴 Sorry, the parking lot is currently **full**. No empty spaces available right now."
            elif self.last_empty_count < self.last_total * 0.15:
                return f"🟠 Limited availability! Only **{self.last_empty_count} spaces** available out of {self.last_total} total. I recommend acting quickly!"
            else:
                return f"🟢 Good news! There are **{self.last_empty_count} empty parking spaces** available out of {self.last_total} total spaces. Would you like me to recommend the best spots?"
        
        # Recommendations
        elif any(word in message_lower for word in ['recommend', 'best', 'top', 'suggest', 'where should', 'which spot']):
            if self.last_empty_count == 0:
                return "❌ No spaces to recommend since the lot is full."
            
            lines = self.last_recommendations.split('\n')
            rec_section = []
            capturing = False
            
            for line in lines:
                if '🥇 RECOMMENDATION #1' in line:
                    capturing = True
                if capturing:
                    rec_section.append(line)
                    if line.strip() == '' and len(rec_section) > 5:
                        break
            
            if rec_section:
                return "🥇 **My top recommendation:**\n\n" + '\n'.join(rec_section[:10]) + "\n\n💡 This spot scored highest based on proximity to entrance, easy access, and detection confidence!"
            return f"I found {self.last_empty_count} good spots! Check the **Image Analysis** tab to see all recommendations with rankings."
        
        # Occupancy rate
        elif any(word in message_lower for word in ['occupancy', 'occupied', 'full', 'busy', 'crowded', 'capacity']):
            if self.last_total > 0:
                occupancy = (self.last_occupied_count / self.last_total) * 100
                
                if occupancy < 30:
                    status = "🟢 Very quiet! Perfect time to park."
                elif occupancy < 60:
                    status = "🟡 Moderately busy but plenty of space."
                elif occupancy < 85:
                    status = "🟠 Getting crowded. Grab a spot soon!"
                else:
                    status = "🔴 Very crowded! Limited options."
                
                return f"📊 **Current Parking Lot Status:**\n\n**Occupancy Rate:** {occupancy:.1f}%\n\n{status}\n\n**Breakdown:**\n• Occupied: {self.last_occupied_count} spaces\n• Available: {self.last_empty_count} spaces\n• Total: {self.last_total} spaces"
            return "I need to analyze an image first to check occupancy."
        
        # Model/Accuracy info
        elif any(word in message_lower for word in ['model', 'accuracy', 'precision', 'recall', 'performance']):
            return f"""🤖 **Model Performance Metrics:**

**Architecture:** {model_info['model_name']} ({model_info['model_variant']})
**Training:** {model_info['epochs_trained']} epochs on {model_info['dataset_size']['train']} images

**Overall Accuracy:**
• mAP50: {model_info['final_metrics']['mAP50']*100:.1f}%
• mAP50-95: {model_info['final_metrics']['mAP50-95']*100:.1f}%
• Precision: {model_info['final_metrics']['precision']*100:.1f}%
• Recall: {model_info['final_metrics']['recall']*100:.1f}%

**Per-Class Performance:**
• Car Detection: {model_info['class_metrics']['car']['mAP50']*100:.1f}% mAP50
• Empty Space Detection: {model_info['class_metrics']['free']['mAP50']*100:.1f}% mAP50

**Speed:** {model_info['inference_speed']} per image"""
        
        # Greeting
        elif any(word in message_lower for word in ['hi', 'hello', 'hey', 'greet']):
            return f"👋 Hello! I analyzed the parking lot and found **{self.last_empty_count} available spaces** out of {self.last_total} total. What would you like to know? I can help with availability, recommendations, occupancy stats, or technical details about the AI system!"
        
        # Help
        elif any(word in message_lower for word in ['help', 'what can']):
            return """💬 **I can help you with:**

🅿️ **Parking Info:**
• Check space availability
• Get personalized spot recommendations
• View occupancy statistics

🤖 **Technical Details:**
• Model training information
• Detection accuracy metrics
• Dataset composition
• How the AI detection works

📊 **Analysis:**
• Current lot capacity
• Best parking strategies
• Real-time availability

Just ask me naturally! Examples:
• "How many spaces are available?"
• "What's the best spot?"
• "How was the model trained?"
• "What's the accuracy?"
• "How does detection work?"
"""
        
        # Default
        else:
            return f"""🤔 I can help you with information about:

**Current Status:** {self.last_empty_count} spaces available out of {self.last_total}

**Ask me about:**
• Parking availability and recommendations
• Occupancy statistics
• Model training details ({model_info['epochs_trained']} epochs, {model_info['final_metrics']['mAP50']*100:.1f}% accuracy)
• How the detection system works
• Dataset information

What would you like to know?"""

    def analyze_video(self, video_path, confidence_threshold=0.4,
                      show_zones=False, show_recommendations=False,
                      entrance_point_coords=None, process_every_n_frames=10,
                      use_aggressive_detection=False, iou_threshold=0.5):
        """
        Optimized video processing with better performance for moving cameras
        """
        import tempfile
        from collections import deque

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"🎥 Video properties - FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {total_frames}")
        
        if not cap.isOpened():
            print("❌ Error: Could not open video stream or file.")
            return None, "❌ Error: Could not open video stream or file."
        
        if fps == 0 or width == 0 or height == 0 or total_frames == 0:
            print(f"⚠️ Warning: One or more video properties are zero. FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {total_frames}")

        # Adjust parameters for aggressive mode
        if use_aggressive_detection:
            adjusted_confidence = max(0.25, confidence_threshold - 0.15)
            adjusted_iou = min(iou_threshold, 0.3)
            max_detections = 300
            agnostic_nms = True
            print(f"🔍 Video: Aggressive detection enabled")
        else:
            adjusted_confidence = confidence_threshold
            adjusted_iou = iou_threshold
            max_detections = 200
            agnostic_nms = False

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0
        
        # Cache for temporal smoothing
        detection_cache = deque(maxlen=3)
        current_annotated_frame = None

        effective_fps = fps / process_every_n_frames if process_every_n_frames > 0 else 0
        print(f"🎬 Processing video: {total_frames} frames at {fps} FPS. Analyzing every {process_every_n_frames} frames (Effective Analysis FPS: {effective_fps:.2f})")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process only selected frames
            if frame_count % process_every_n_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save temp frame
                temp_frame_path = 'temp_video_frame.jpg'
                cv2.imwrite(temp_frame_path, frame)
                
                # Run detection with optimized settings
                results = self.model.predict(
                    temp_frame_path,
                    conf=adjusted_confidence,
                    iou=adjusted_iou,
                    verbose=False,
                    agnostic_nms=use_aggressive_detection,
                    max_det=max_detections
                )[0]

                boxes = results.boxes
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                empty_spaces = []
                occupied_spaces = []

                # Simplified area filtering for speed
                all_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in xyxy]
                if len(all_areas) > 0:
                    median_area = np.median(all_areas)
                    if use_aggressive_detection:
                        min_area = median_area * 0.2
                        max_area = median_area * 4.0
                    else:
                        min_area = median_area * 0.3
                        max_area = median_area * 3.0
                    
                    median_width = np.median([box[2] - box[0] for box in xyxy])
                    if median_width > 0:
                        self.pixels_per_meter = median_width / 2.5
                    else:
                        self.pixels_per_meter = 50
                else:
                    min_area, max_area = 0, float('inf')
                    self.pixels_per_meter = 50

                for box, cls, conf in zip(xyxy, classes, confidences):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area < min_area or area > max_area:
                        continue

                    space_info = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        'area': float(area),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1]),
                        'id': None
                    }

                    if cls == 1:
                        empty_spaces.append(space_info)
                    elif cls == 0:
                        occupied_spaces.append(space_info)

                all_spaces = empty_spaces + occupied_spaces
                self.assign_spot_ids(all_spaces, width, height)
                
                # Store in cache
                detection_cache.append((empty_spaces, occupied_spaces))

                # Simplified visualization for video (faster rendering)
                annotated_image_pil = self.create_enhanced_visualization(
                    frame_rgb, empty_spaces, occupied_spaces,
                    show_zones=show_zones,
                    show_recommendations=show_recommendations
                )

                current_annotated_frame = cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR)

                if current_annotated_frame.shape[0] != height or current_annotated_frame.shape[1] != width:
                    print(f"Resizing frame from {current_annotated_frame.shape[1]}x{current_annotated_frame.shape[0]} to {width}x{height}")
                    current_annotated_frame = cv2.resize(current_annotated_frame, (width, height), interpolation=cv2.INTER_AREA)

                processed_count += 1

                print(f"Frame {frame_count}: Original frame shape={frame.shape}, dtype={frame.dtype}")
                if current_annotated_frame is not None:
                    print(f"Frame {frame_count}: Annotated frame shape={current_annotated_frame.shape}, dtype={current_annotated_frame.dtype}")
                else:
                    print(f"Frame {frame_count}: Annotated frame not yet generated, writing original frame.")

            # Write frame
            out.write(current_annotated_frame if current_annotated_frame is not None else frame)

            # Progress update every 2 seconds
            if frame_count % (fps * 2) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        cap.release()
        out.release()

        print(f"✅ Video processing complete! Processed {processed_count} frames out of {total_frames}")

        stats = f"""
🎬 VIDEO ANALYSIS COMPLETE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 PROCESSING SUMMARY:
   Frames Processed:      {processed_count}/{total_frames}
   Processing Rate:       Every {process_every_n_frames} frames
   Total Duration:        {total_frames/fps:.1f} seconds
   Effective Analysis:    {processed_count/(total_frames/fps):.1f} detections/sec
   
   Aggressive Mode:       {'✅ Enabled' if use_aggressive_detection else '❌ Disabled'}
   Zones Displayed:       {'✅ Yes' if show_zones else '❌ No'}
   Recommendations:       {'✅ Yes' if show_recommendations else '❌ No'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 OPTIMIZATION TIPS:
   • Increase "Process Every N Frames" (15-30) for faster processing
   • Disable zones/recommendations for maximum speed
   • Enable aggressive detection for crowded or moving camera footage
   • For detailed analysis, extract key frames as images

⚡ PERFORMANCE NOTES:
   • Video processing is slower than image analysis
   • Moving cameras require more frequent frame processing
   • Complex scenes benefit from aggressive detection mode
"""

        return output_path, stats


def create_gradio_app(model_path, model_path_2=None):
    """Create Gradio interface with ENHANCED controls for crowded lots and videos"""
    detector = EnhancedParkingDetectionGUI(model_path, model_path_2)

    with gr.Blocks(title="Enhanced Smart Parking System", theme=gr.themes.Default()) as app:
        gr.Markdown("""
        # 🚗 Enhanced Smart Parking Detection System
        
        **AI-Powered Parking Space Detection with Intelligent Recommendations**
        
        ✨ **New Features:**
        - 🔍 Aggressive Detection Mode for crowded/unsymmetric parking lots
        - ⚡ Optimized video processing for moving cameras
        - 🎯 Improved detection accuracy
        
        Upload an image or video and click to set entrance location!
        """)

        entrance_coords_state = gr.State(value=[])

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════
            # IMAGE ANALYSIS TAB
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("📷 Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Image")
                        gr.Markdown("👆 **Click on the image to set entrance location**")
                        input_image = gr.Image(type="numpy", label="Parking Lot Image")

                        gr.Markdown("### ⚙️ Detection Settings")
                        img_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.4,
                            step=0.05,
                            label="🎯 Detection Confidence",
                            info="Lower = more detections | Higher = only confident"
                        )
                        
                        img_iou = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="🎯 IOU Threshold (NMS)",
                            info="Lower = fewer overlaps | Higher = more overlaps allowed (0.5 recommended)"
                        )
                        
                        img_aggressive = gr.Checkbox(
                            value=False,
                            label="🔍 Enable Aggressive Detection Mode",
                            info="Recommended for: crowded lots, unsymmetric layouts, complex scenes"
                        )

                        gr.Markdown("### 🎨 Visualization Options")
                        img_show_zones = gr.Checkbox(
                            value=True,
                            label="Show Zone Analysis Grid",
                            info="Divide lot into color-coded zones"
                        )

                        img_show_recs = gr.Checkbox(
                            value=True,
                            label="Show Top 5 Recommendations",
                            info="Highlight and rank best spots"
                        )

                        img_detect_btn = gr.Button(
                            "🔍 Analyze Image",
                            variant="primary",
                            size="lg"
                        )

                        gr.Markdown("### 📊 Analysis Report")
                        img_stats_output = gr.Textbox(
                            label="Detailed Statistics",
                            lines=24,
                            max_lines=30
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📸 Annotated Results")
                        img_output = gr.Image(label="Detection Visualization", type="pil")

                        img_entrance_status = gr.Markdown("📍 **Entrance:** Not set (Using default bottom-center)")

                        gr.Markdown("### 🎯 Smart Recommendations")
                        img_recs_output = gr.Textbox(
                            label="Top Parking Spots Ranked",
                            lines=28
                        )

                def on_image_select(evt: gr.SelectData):
                    if evt.index is None:
                        return None, "📍 **Entrance:** Click on image to set location"
                    
                    coords = [evt.index[0], evt.index[1]]
                    status_msg = f"📍 **Entrance Set At:** X={coords[0]}, Y={coords[1]}"
                    return coords, status_msg

                input_image.select(
                    fn=on_image_select,
                    inputs=None,
                    outputs=[entrance_coords_state, img_entrance_status]
                )

                img_detect_btn.click(
                    fn=detector.analyze_and_visualize,
                    inputs=[input_image, img_confidence, img_show_zones, img_show_recs, 
                            entrance_coords_state, img_aggressive, img_iou],
                    outputs=[img_output, img_stats_output, img_recs_output]
                )

            # ═══════════════════════════════════════════════════════════
            # VIDEO ANALYSIS TAB
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("🎥 Video Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Video")
                        gr.Markdown("⚠️ **Note**: Video processing takes time. Optimize settings below.")
                        input_video = gr.Video(label="Parking Lot Video")

                        gr.Markdown("### ⚙️ Video Processing Settings")
                        vid_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.4,
                            step=0.05,
                            label="🎯 Detection Confidence"
                        )

                        vid_iou = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="🎯 IOU Threshold",
                            info="Lower = fewer overlaps | Higher = more overlaps (0.5 recommended)"
                        )

                        process_rate = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=10,
                            step=1,
                            label="⏭️ Process Every N Frames",
                            info="Higher = faster (10-15 recommended, 20-30 for moving camera)"
                        )
                        
                        vid_aggressive = gr.Checkbox(
                            value=False,
                            label="🔍 Enable Aggressive Detection",
                            info="Use for crowded lots or moving camera"
                        )

                        vid_show_zones = gr.Checkbox(
                            value=False,
                            label="Show Zone Analysis (slower)",
                            info="Disable for faster processing"
                        )

                        vid_show_recs = gr.Checkbox(
                            value=False,
                            label="Show Recommendations (slower)",
                            info="Disable for faster processing"
                        )

                        vid_detect_btn = gr.Button(
                            "🎬 Process Video",
                            variant="primary",
                            size="lg"
                        )

                        vid_stats_output = gr.Textbox(
                            label="Processing Status",
                            lines=18
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 🎬 Processed Video")
                        vid_output = gr.Video(label="Annotated Video")

                        gr.Markdown("""
                        ### 💡 Video Processing Tips:
                        
                        **For Best Performance:**
                        - 🚀 Set "Process Every N Frames" to 15-30 for fast results
                        - ⚡ Disable zones and recommendations
                        - 🔍 Enable aggressive detection for complex scenes
                        
                        **For Best Quality:**
                        - 🎯 Set "Process Every N Frames" to 5-10
                        - ✅ Enable zones and recommendations
                        - 📹 Works best with static camera footage
                        
                        **For Moving Camera:**
                        - 🎥 Use higher frame skip rate (20-30)
                        - 🔍 Enable aggressive detection
                        - ⚡ Expect slower but more accurate results
                        
                        **Important:** Long videos (>1 min) may take several minutes to process.
                        For detailed analysis, extract key frames as images instead.
                        """)

                def process_video_wrapper(video, confidence, zones, recs, rate, aggressive, iou):
                    if video is None:
                        return None, "⚠️ Please upload a video first"

                    cap = cv2.VideoCapture(video)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    output_path, stats = detector.analyze_video(
                        video, confidence, zones, recs, None, int(rate), aggressive, iou
                    )

                    return output_path, stats

                vid_detect_btn.click(
                    fn=process_video_wrapper,
                    inputs=[input_video, vid_confidence, vid_show_zones, vid_show_recs,
                           process_rate, vid_aggressive, vid_iou],
                    outputs=[vid_output, vid_stats_output]
                )

            # ═══════════════════════════════════════════════════════════
            # CHATBOT TAB
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("💬 Parking Assistant Chatbot"):
                gr.Markdown("""
                ### 🤖 Chat with Your AI Parking Assistant (Powered by Hugging Face)

                **First:** Analyze an image in the "Image Analysis" tab
                **Then:** Come back here and ask me anything!
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Ask Me About:

                        **Parking Info:**
                        - *"How many spaces available?"*
                        - *"Recommend a spot"*
                        - *"What's the occupancy?"*

                        **Model Training:**
                        - *"How was the model trained?"*
                        - *"What's the accuracy?"*
                        - *"How many epochs?"*
                        - *"What dataset was used?"*

                        **Technical Details:**
                        - *"What technology is used?"*
                        - *"How does detection work?"*
                        - *"Show me the tech stack"*

                        **System Features:**
                        - *"What is aggressive detection?"*
                        - *"How to improve detection?"*
                        - *"Tips for crowded lots?"*

                        Type **'help'** for all options!
                        """)

                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Parking Assistant (Hugging Face-Powered)",
                            height=500,
                            type='messages'
                        )

                        with gr.Row():
                            msg = gr.Textbox(
                                label="Your Message",
                                placeholder="Ask me about parking availability, recommendations, or anything else...",
                                lines=2,
                                scale=4
                            )
                            submit_btn = gr.Button("Send", scale=1, variant="primary")

                        with gr.Row():
                            clear_btn = gr.Button("Clear Chat", scale=1)
                            retry_btn = gr.Button("Retry Last", scale=1)

                def respond(message, chat_history):
                    if not message.strip():
                        return chat_history, ""
                    
                    tuples_history = []
                    if isinstance(chat_history, list):
                        for item in chat_history:
                            if isinstance(item, dict):
                                if item.get('role') == 'user':
                                    user_msg = item.get('content', '')
                                    continue
                            elif isinstance(item, tuple):
                                 tuples_history.append(item)

                    bot_response = detector.chat_with_bot(message, tuples_history)
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": bot_response})
                    return chat_history, ""
           
                def retry_last(chat_history):
                    if not chat_history or len(chat_history) == 0:
                        return chat_history, ""

                    if len(chat_history) > 0:
                        last_msg, _ = chat_history.pop()
                        return respond(last_msg, chat_history)
                    
                    return chat_history, ""
                
                msg.submit(respond, [msg, chatbot], [chatbot, msg])
                submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
                clear_btn.click(lambda: [], None, chatbot)
                retry_btn.click(retry_last, [chatbot], [chatbot, msg])

    return app

if __name__ == "__main__":
    model_path = "best.pt"
    model_path_2 = "best2.pt"  # Your second model
    
    # Check if first model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
    else:
        # Check if second model exists
        if not os.path.exists(model_path_2):
            print(f"⚠️ Warning: Second model not found at {model_path_2}")
            print(f"ℹ️ Running with single model only")
            app = create_gradio_app(model_path, None)
        else:
            print(f"🔀 Running in ENSEMBLE mode with both models")
            app = create_gradio_app(model_path, model_path_2)
        
        app.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            show_api=False
        )