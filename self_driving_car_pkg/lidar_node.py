import numpy as np
import os
import sys
import cv2
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from my_msgs.msg import Objects, Tracker
import time
import random

# print문 buffer 비활성화 -> print문 바로 출력
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

class LidarTracker(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar/lidar_scan',
            self.lidar_callback,
            10
        )
        self.publisher = self.create_publisher(Objects, '/tracked_objects', 10)
        timer_period = 0.03;self.timer = self.create_timer(timer_period, self.process_func)
        self.previous_centroids = []  # 이전 프레임의 객체 중심 좌표
        self.epsilon = 0.3  # DBSCAN 클러스터링의 최대 거리 (m)
        self.tracking_thres = 0.5
        self.points = None
        self.last_id = 0


    def lidar_callback(self, msg):
        # 1. LiDAR 데이터 변환
        self.points = self.convert_laserscan_to_points(msg)

        

        

    def convert_laserscan_to_points(self, msg):
        """LaserScan 데이터를 x, y 좌표로 변환"""
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # 유효한 거리 데이터만 필터링
        valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            # self.get_logger().error("No valid ranges found. All ranges are either too close or too far.")
            return np.array([])  # 비어있는 배열 반환

        # 극좌표 -> 직교좌표 변환
        x = ranges * -np.sin(angles)
        y = ranges * np.cos(angles)
        return np.vstack((x, y)).T
    

    def process_func(self):
        if self.points is not None:
            start_t = time.time()
            # 2. 클러스터링 (객체 감지)
            centroids = self.detect_objects(self.points)
            end_t = time.time()
            print(f'클러스터링 소요 시간 : {end_t - start_t}')
            # 3. 객체 트래킹
            curr_centroids = self.track_objects(centroids)

            # 4. 결과 발행
            self.publish_tracked_objects(curr_centroids)

            # 이전 프레임 중심 업데이트
            self.previous_centroids = curr_centroids
            end_t = time.time()
            print(f'전체 소요 시간 : {end_t - start_t}')

    def detect_objects(self, points):
        """DBSCAN을 사용하여 객체 감지"""
        if points.shape[0] == 0:
            # self.get_logger().warning("No points to cluster. Skipping clustering step.")
            return []

        clustering = DBSCAN(eps=self.epsilon, min_samples=3).fit(points)
        labels = clustering.labels_

        # 각 클러스터의 중심 계산
        centroids = []
        for label in set(labels):
            if label == -1:  # 노이즈는 무시
                continue
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            # print(f'label: {label}, point: {centroid}')
            centroids.append(centroid)
            

        return centroids

    def track_objects(self, centroids):
        """이전 프레임의 중심과 현재 중심을 매칭하여 객체를 추적"""
        curr_centroids = []

        if len(centroids) == 0: # 객체가 없는 경우에는 이전 중심을 초기화하고, 빈 값 return
            self.previous_centroids = []
            return []
        if len(self.previous_centroids) == 0: # 이전 객체가 없던 경우
            for curr in centroids:
                curr_centroids.append((curr[0],curr[1], self.last_id+1)) # 현재 중심만 추가
            return curr_centroids
        pair_lst = []
        
        print(f'centroids: {centroids}')
        # 이전 중심과 현재 중심 매칭
        for prev in (self.previous_centroids):
            prev_x, prev_y, prev_id = prev
            for i, curr in enumerate(centroids):
                curr_x, curr_y = curr
                distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5 # 중심끼리의 거리 계산
                print(f'{prev}, {curr} distance : {distance}')
                if distance <= self.tracking_thres: # 거리가 임계치보다 작은 경우 같은 객체로 인식
                    curr_centroids.append((curr_x,curr_y, prev_id)) # 이전 객체와 같은 id 부여
                    pair_lst.append(i) # 매칭된 현재 중심의 인덱스 저장
                    if prev_id > self.last_id:
                        self.last_id = prev_id # 부여된 id 중 가장 큰 값 저장
                    break
        cnt = 0
        pair_lst.sort() # 인덱스 정렬
        print(pair_lst)
        # 매칭 성공한 중심점 제거
        for i in set(pair_lst):
            print(i)
            print(centroids, i -cnt)
            del centroids[i -cnt]
            cnt +=1
        
        
        if centroids != []: # 매칭되지 않은 객체가 존재하는 경우(새로운 객체)
            print("!!!!!!!!!!!!!!!!  new object !!!!!!!!!!!!!!!!!!")
            add_id = 1
            # 새로운 id 생성
            for new in centroids:
                new_x, new_y = new
                curr_centroids.append((new_x,new_y, self.last_id + add_id))
                add_id += 1
        print(f'current result: {curr_centroids}')

        return curr_centroids

    def publish_tracked_objects(self, tracked_objects):
        """트래킹된 객체의 좌표를 퍼블리시"""
        # if len(tracked_objects) == 0:
        #     # self.get_logger().warning("No tracked objects to publish.")
        #     return
        objects = Objects()
        for obj in tracked_objects:
            tracker = Tracker()
            tracker.x = float(obj[0])
            tracker.y = float(obj[1])
            tracker.id = int(obj[2])
            objects.objects.append(tracker)
        self.publisher.publish(objects)
        

def main(args=None):
    rclpy.init(args=args)
    tracker = LidarTracker()
    rclpy.spin(tracker)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
