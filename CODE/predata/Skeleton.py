# -*- encoding: utf-8 -*-
# @File    :   Skeleton.py
# @Time    :   2023/11/14 09:57:57
# @Author  :   
# @Contact :   
# @Desc    :   None

import time
import pygeos
import numpy as np
from collections import defaultdict
"""
没写完，不知道后面还会不会写
"""
class Skeleton:
    def __init__(self):
        self.triangle_data = None
        self.lines = None
        self.visual_dist_unit = None

    def load_triangle_data(self,triangle_data):
        self.triangle_data = triangle_data

    def analysis_triangle_data(self):
        # print(f"TIME:{time.ctime()}----> ANALYSIS TRIANGLE DATA")
        tri_s = self.triangle_data['tri_s']
        tri_s_neighbors = tri_s.neighbors
        building_in_triangle = self.triangle_data['del_index'] # 在建筑物内部的三角形的索引
        tri_s_neighbors[np.isin(tri_s_neighbors, building_in_triangle)] = -2 # 将建筑物内部的三角形的索引置为-1
        tri_s_neighbors[building_in_triangle] = -2
        triangle_vertices = self.triangle_data['triangle_vertices'] # 三角形的三个顶点的坐标
        # triangle_vertices_del = np.delete(self.triangle_data['triangle_vertices'],building_in_triangle,axis=0) # self.triangle_data['triangle_vertices'] # 三角形的三个顶点的坐标
        tri_s_judge_index = np.where(tri_s_neighbors >=0 ,1,0) # 顶点对应的边有邻接三角形，为1，否则为0,  tri_s_judge_index为三角形index
        # tri_s_judge_index[building_in_triangle] = -1 # 将建筑物内部的三角形的索引置为-1
        # tri_s_judge_index_del = np.delete(tri_s_judge_index,building_in_triangle,axis=0)
        tri_s_judge_index_sum = np.sum(tri_s_judge_index,axis=1) # 顶点对应的边有邻接三角形的个数
        # tri_s_judge_index_sum_del = np.delete(tri_s_judge_index_sum,building_in_triangle,axis=0)
        # lines_from_3, lines_3s = self.create_line_with_3_neighbor(triangle_vertices_del[np.where(tri_s_judge_index_sum_del == 3)],np.where(tri_s_judge_index_sum_del == 3)[0])
        # lines_from_2, lines_2s = self.create_line_with_2_neighbor(triangle_vertices_del[np.where(tri_s_judge_index_sum_del == 2)],tri_s_judge_index_del[np.where(tri_s_judge_index_sum_del == 2)],np.where(tri_s_judge_index_sum_del == 2)[0])

        lines_from_3, lines_3s = self.create_line_with_3_neighbor(triangle_vertices[np.where(tri_s_judge_index_sum == 3)],np.where(tri_s_judge_index_sum == 3)[0])
        lines_from_2, lines_2s = self.create_line_with_2_neighbor(triangle_vertices[np.where(tri_s_judge_index_sum == 2)],tri_s_judge_index[np.where(tri_s_judge_index_sum == 2)],np.where(tri_s_judge_index_sum == 2)[0])

        # 骨架线对映到多边形，多边形对映的骨架线
        triangle_node_from_polygon = self.triangle_data['triangle_node_from_polygon'] # np.delete(self.triangle_data['triangle_node_from_polygon'],building_in_triangle,axis=0) # 三角形的三个顶点来自哪个多边形
        # triangle_node_from_polygon = np.delete(self.triangle_data['triangle_node_from_polygon'],building_in_triangle,axis=0) # 三角形的三个顶点来自哪个多边形
        skeleton_2_polygon = defaultdict(list)
        polygon_2_skeleton = defaultdict(list)
        skeleton_from_triangle = []
        # 连接三个建筑物的三角形的骨架线
        triangle_3s = triangle_node_from_polygon[lines_3s[1]] #
        for iter in range(triangle_3s.shape[0]):
            skeleton_2_polygon[iter*3].extend([triangle_3s[iter,1], triangle_3s[iter,2]])
            skeleton_2_polygon[iter*3+1].extend([triangle_3s[iter,0], triangle_3s[iter,2]])
            skeleton_2_polygon[iter*3+2].extend([triangle_3s[iter,1], triangle_3s[iter,0]])
            skeleton_from_triangle.extend([lines_3s[1][iter] for _ in range(3)])

        # 骨架线个数
        skeleton_num_now = len(lines_3s[1]) * 3
        # 连接两个建筑物的三角形的骨架线
        triangle_2s = triangle_node_from_polygon[lines_2s]
        for iter in range(triangle_2s.shape[0]):
            skeleton_2_polygon[skeleton_num_now + iter].extend(list(set(triangle_2s[iter])))
            skeleton_from_triangle.append(lines_2s[iter])
        skeletons = np.vstack([lines_from_3,lines_from_2])
        # skeletons = lines_from_2
        for skeleton_idx in skeleton_2_polygon.keys():
            if len(skeleton_2_polygon[skeleton_idx]) >= 2: # 一个骨架线至少连接两个多边形
                for polygon_idx in skeleton_2_polygon[skeleton_idx]:
                    polygon_2_skeleton[polygon_idx].append(skeleton_idx)
        self.skeleton_2_polygon = skeleton_2_polygon
        self.polygon_2_skeleton = polygon_2_skeleton
        self.lines = skeletons
        self.skeleton_from_triangle = skeleton_from_triangle

        # print(f"TIME:{time.ctime()}----> ANALYSIS TRIANGLE DATA SUCCESS")

    def create_line_with_1_neighbor(self):
        """
        在这里，只有一个邻居三角形的 在构建骨架线时，最后并不能作为构建多边形的边，可以不生成
        """
        pass

    def create_line_with_2_neighbor(self,triangle_vertices,tri_s_judge_index,tri_s_index):
        line_mid_point = np.mean([[triangle_vertices[:,1,:],triangle_vertices[:,2,:]],
                            [triangle_vertices[:,2,:],triangle_vertices[:,0,:]],
                            [triangle_vertices[:,0,:],triangle_vertices[:,1,:]]],axis=1).transpose(1,0,2) #三角形每条边的中点，按序分别对应三角形的三个顶点
        tri_s_judge_index_flatten = tri_s_judge_index.flatten()
        line_mid_point_flatten = line_mid_point.reshape(-1,2)
        line_mid_point_flatten = line_mid_point_flatten[np.where(tri_s_judge_index_flatten == 1)]
        return line_mid_point_flatten.reshape(-1,2,2), tri_s_index


    def create_line_with_3_neighbor(self,triangle_vertices,tri_s_index):
        centroids = np.mean(triangle_vertices,axis=1) # 三角形的质心
        line_mid_point = np.mean([[triangle_vertices[:,2,:],triangle_vertices[:,1,:]],
                                  [triangle_vertices[:,0,:],triangle_vertices[:,2,:]], 
                                  [triangle_vertices[:,1,:],triangle_vertices[:,0,:]]] 
                                  ,axis=1).transpose(1,0,2) #三角形每条边的中点
        
        lines = np.stack([line_mid_point,np.asarray([centroids for i in range(3)]).transpose(1,0,2)],axis=2) # 三角形每条边的中点和质心连线
        
        return lines.reshape(-1,2,2),[lines,tri_s_index] # （线段的数量，线段的两个端点，每个端点的坐标）,[三条线，三角形的索引]

    def delete_lines_overlap_buildings(self,building_data_tree):
        # print(f"TIME:{time.ctime()}----> DELETE LINES OVERLAP BUILDINGS")
        lines = [pygeos.linestrings(line) for line in self.lines]
        lines_overlap_buildings = building_data_tree.query_bulk(lines, predicate='intersects')
        # 删除查询出来的lines
        # self.lines = np.delete(self.lines,lines_overlap_buildings[0,:],axis=0)
        self.overlap_buildings_lines = lines_overlap_buildings[0,:]

    def get_result(self):
        return {
            "skeletons": self.lines,
            "skeleton_2_polygon": self.skeleton_2_polygon,
            "polygon_2_skeleton": self.polygon_2_skeleton,
            "skeleton_from_triangle": self.skeleton_from_triangle,
            "overlap_buildings_lines": self.overlap_buildings_lines
        }