# -*- encoding: utf-8 -*-
# @File    :   BuildingDataloader.py
# @Time    :   2023/11/13 09:22:40
# @Author  :   
# @Contact :   
# @Desc    :   None
import collections
import math
import os
import shapefile
import scipy
import numpy as np
import shapely.geometry as geometry
import shapely
import tqdm
import matplotlib.pyplot as plt
import time
import pygeos
from scipy.spatial import Delaunay # 虽然 pygeos 也有构建三角网的方法，但是这个之前用过
# import triangle
from Skeleton import Skeleton
from sklearn.preprocessing import StandardScaler, MinMaxScaler

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class BuildingDataLoader:
    def __init__(self,labels,png_path=None):
        self.shapes = None
        self.tree = None
        self.shp_file = None
        self.pygoes_polygons = None
        self.skeleton = None
        self.tri = None
        self.attrs = None
        self.attrs_2_building = None

        self.labels = labels
        self.building_label = None
        self.building_type = None
        self.file_path = None
        self.file_name = None
        self.global_attrs = None
        self.buildings_centroid = None

        self.png_path = png_path

    def get_buildings_centroid(self):
        """
        Get buildings' centroid
        """
        if self.shapes is None:
            self.shapes = self.shp_file.shapes()
        return geometry.MultiPolygon([geometry.Polygon(i.points) for i in self.shapes]).centroid

    def get_label(self):
        """
        从父级目录中获取数据标签
        """
        self.building_type = os.path.basename(os.path.dirname(self.file_path))
        self.building_label = self.labels[self.building_type]

    def buildings_rectangularity(self,buildings: list):
        """
        多建筑物的矩形度
        """
        multi_polygon = geometry.MultiPolygon([geometry.Polygon(i.points) for i in buildings])
        hull_area = multi_polygon.convex_hull.area
        rect_area = multi_polygon.minimum_rotated_rectangle.area
        return hull_area/rect_area

    def load_data_from_shp(self,shp_path):
        """
        Load data from shapefile
        """
        self.file_path = shp_path
        self.get_label()
        self.file_name = '.'.join((shp_path.split('.')[:-1]))
        self.shp_file = shapefile.Reader(shp_path)
        # 如果有MultiPolygon，就转换为Polygon

    def get_building_individual_attr(self,building_points):
        """
        Get individual attributes of a building
        """
        if self.buildings_centroid is None:
            self.buildings_centroid = self.get_buildings_centroid()
        b = geometry.Polygon(building_points).buffer(0)
        centroid = b.centroid # 中心点
        area = b.area # 面积
        sbr = b.minimum_rotated_rectangle # 最小外接矩形
        angle_and_length = self.get_rectangle_angle_and_length(sbr)
        sbr_area = sbr.area # 最小外接矩形面积
        stretch = angle_and_length['short_length']/angle_and_length['long_length'] # 1/延展度
        rectangularity = area/sbr_area # 矩形度
        # BGCO
        # bgco = self.get_angle_and_length([[centroid.x,centroid.y],[self.buildings_centroid.x,self.buildings_centroid.y]])['angle_radians']
        #先化简，再计算边数
        # new_b = b.simplify(0.5,preserve_topology=True)
        # if isinstance(new_b,shapely.geometry.MultiPolygon):
        #     print()
        # building_points = list(new_b.exterior.coords)
        edge_number = len(building_points) if building_points[0] != building_points[-1] else len(building_points) + 1# 边数
        # fenxing = 2. * math.log(b.length) / math.log(b.area) if b.area != 0 else 0
        return {
            'area':area,
            'centroid':[centroid.x,centroid.y],
            'sbr_angle':angle_and_length['angle_radians'],
            'stretch':stretch,
            # 'bgco':bgco,
            'rectangularity':rectangularity,
            'edge_number':edge_number
        }

    def get_rectangle_angle_and_length(self,min_rect):
        """
        Get the angle of the rectangle, and long edge and short edge 's length
        """
        # 转换为4个点的坐标
        coords = list(min_rect.exterior.coords)

        # 从矩形的四个点中计算两条相邻边的向量
        edge1 = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        edge2 = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])

        length1 = math.hypot(*edge1)
        length2 = math.hypot(*edge2)

        long_length = length1 if length1 >= length2 else length2
        short_length = length1 if length1 < length2 else length2

        # 选择较长的边用于计算方向
        edge = edge1 if length1 >= length2 else edge2

        # 计算边的角度（与x轴的夹角），使用atan2保证方向正确
        angle_radians = math.atan2(edge[1], edge[0])
        angle_degrees = math.degrees(angle_radians)

        # 确保角度位于第一或第二象限
        if angle_degrees < 0:
            angle_degrees += 180
        elif angle_degrees > 180:
            angle_degrees -= 180
        if angle_radians < 0:
            angle_radians += math.pi
        elif angle_radians > math.pi:
            angle_radians -= math.pi

        # 输出结果
        # print("角度（度）：", angle_degrees)
        # print("弧度：", angle_radians)
        return {'angle_degrees':angle_degrees,'angle_radians':angle_radians,'long_length':long_length,'short_length':short_length}

    def get_angle_and_length(self,edge):
        # 计算边的角度（与x轴的夹角），使用atan2保证方向正确
        edge = (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])
        angle_radians = math.atan2(edge[1], edge[0])
        angle_degrees = math.degrees(angle_radians)
        length = math.hypot(*edge)
        # 确保角度位于第一或第二象限
        if angle_degrees < 0:
            angle_degrees += 180
        elif angle_degrees > 180:
            angle_degrees -= 180
        if angle_radians < 0:
            angle_radians += math.pi
        elif angle_radians > math.pi:
            angle_radians -= math.pi
        return {'angle_degrees':angle_degrees,'angle_radians':angle_radians,'length':length}
    
    def write_common_edge_building(self):
        # print(f'TIME:{time.ctime()}----> START WRITE COMMON EDGE BUILDING')
        self.create_rtree()
        querys = self.tree.query_bulk(self.pygoes_polygons,predicate='touches').T.tolist()
        self.calculate_attrs()
        if self.attrs_2_building is None:
            self.attrs_2_building = {}
        for all_idx,(i_idx,j_idx) in enumerate(querys): # TODO 可以优化，感觉现在没必要
            if tuple(sorted((i_idx,j_idx))) in self.attrs_2_building.keys():
                continue
            self.attrs_2_building[tuple(sorted((i_idx,j_idx)))] = self.calculate_next2_features(i_idx,j_idx, common_edge=True)
        # print(f'TIME:{time.ctime()}----> END WRITE COMMON EDGE BUILDING')
        return len(querys)

    def create_rtree(self):
        """
        Create rtree
        """
        if self.shapes is None:
            self.shapes = self.shp_file.shapes()
        if self.tree is not None and self.pygoes_polygons is not None:
            return
        polygons = []
        for shape in self.shapes:
            p = pygeos.polygons(shape.points)
            if pygeos.is_valid(p):
                polygons.append(p)
            else:
                polygons.append(pygeos.make_valid(p))
        self.pygoes_polygons = polygons
        if self.tree is None:
            self.tree = pygeos.STRtree(polygons)

    def create_delaunay(self,add_model=False):
        """
        TODO 调用scipy.spatial.Delaunay创建三角网,有待自己实现基于约束的三角网算法
        """
        # print(f'TIME:{time.ctime()}----> START CREATE DELAUNAY')
        self.create_rtree()
        big_rectangle,extend = self.create_big_hull()
        polygons = self.pygoes_polygons
        if len(polygons) > 3:
            polygons.append(pygeos.buffer(pygeos.polygons(big_rectangle),0,cap_style='flat'))

        extend_points = [] # 插值后所有的点
        point_num_each_polygon = []
        extend_points_from_witch_polygon = [] # 每个点来自哪个多边形
        # print(f'TIME:{time.ctime()}---->    START SEGMENTIZE')
        # 对多边形轮廓进行插值, 保险起见，一个个循环插值吧
        for idx,p in enumerate(polygons):
            points = pygeos.get_coordinates(pygeos.segmentize(p,extend)).tolist()
            points = points[:-1] if points[0] == points[-1] else points
            len_points = len(points)
            extend_points.extend(points)
            point_num_each_polygon.append(len_points)
            if idx == len(polygons) - 1 and len(polygons) > 3:
                extend_points_from_witch_polygon.extend([-2] * len_points) # 外边框上的点标记为 -2 index
            else:
                extend_points_from_witch_polygon.extend([idx] * len_points)
        # print(f'TIME:{time.ctime()}---->    END SEGMENTIZE')
        # print(f'TIME:{time.ctime()}---->    START DELAUNAY {len(extend_points)} POINTS')

        tri_s = Delaunay(extend_points,qhull_options='Q12 Qbb')

        # print(f'TIME:{time.ctime()}---->    END DELAUNAY')
        # 三角网的顶点， 方便后续转换为多边形
        self.extend_points = extend_points # TODO
        triangle_vertices = np.asarray(extend_points)[tri_s.simplices]
        triangle_node_from_polygon = np.asarray(extend_points_from_witch_polygon)[tri_s.simplices]
        if_eq_is_1 = (triangle_node_from_polygon[:, 0] == triangle_node_from_polygon[:, 1]) & (triangle_node_from_polygon[:, 1] == triangle_node_from_polygon[:, 2]) # 三角形的三个顶点来自同一个多边形
        del_index = np.where(if_eq_is_1==1)[0].tolist()
        # # 得到三角网，已经删除了多边形内部的三角形
        # triangle_vertices = np.delete(triangle_vertices,del_index,axis=0)
        # print(f'TIME:{time.ctime()}----> END DELAUNAY TRIANGLES, DELETE {len(del_index)} TRIANGLES, REMAIN {len(triangle_vertices)} TRIANGLES')
        # print(f'TIME:{time.ctime()}----> END DELAUNAY TRIANGLES')
        self.tri = {
            'triangle_vertices':triangle_vertices,
            'del_index':del_index,
            'tri_s':tri_s,
            'triangle_node_from_polygon':triangle_node_from_polygon,
        }

    def calculate_attrs(self):
        if self.attrs is not None:
            return
        all_attrs = []
        for idx,building in enumerate(self.shapes):
            all_attrs.append(self.get_building_individual_attr(building.points))
        self.attrs = all_attrs

    def easy_create_next2_by_triangle(self):
        """
        跟创建骨架线得到的结果一样，还简单一些
        """
        # 获取与外边框相连的三角形，删除掉，用这个方法时，不需要那些三角形
        # triangle_index = self.tri['tri_s'].simplices
        del_index = self.tri['del_index']
        triangle_node_from_polygon = self.tri['triangle_node_from_polygon']
        if_on_border = np.isin(triangle_node_from_polygon,-1).sum(axis=1) # 在边框上的三角形为1，不在边框上的三角形为0
        if_on_border_index = np.where(if_on_border == 1)[0]
        del_index.extend(if_on_border_index.tolist())
        all_del_index = list(set(del_index))
        # 删除三角形
        triangle_node_from_polygon_index = np.delete(triangle_node_from_polygon,all_del_index,axis=0)
        # triangle_index = np.delete(triangle_index,all_del_index,axis=0)
        dd = collections.defaultdict(int)
        # print(f'TIME:{time.ctime()}----> START CREATE NEXT2 BY TRIANGLE')
        for building_idx in triangle_node_from_polygon_index:
            building_idx = np.sort(building_idx)
            if building_idx[0] >=0 and building_idx[1] >= 0 and building_idx[2] >= 0:
                if building_idx[0] < building_idx[1] and building_idx[1] < building_idx[2]:
                    # 三角形位于三个不同的建筑物之间
                    # dd[(building_idx[0],building_idx[1])] += 1
                    # dd[(building_idx[1],building_idx[2])] += 1
                    pass
                else:
                    # 三角形位于两个不同的建筑物之间
                    dd[(building_idx[0],building_idx[2])] += 1
        # print(f'TIME:{time.ctime()}----> END CREATE NEXT2 BY TRIANGLE')
        
        # print(f'TIME:{time.ctime()}----> START CALCULATE RELATIONSHIP ATTRS')
        if len(dd) == 0:
            print(f'No EDGE in {self.file_path}')
        connect_threshold = 2
        if max(dd.values()) < 2:
            connect_threshold = 1
        self.calculate_attrs()
        if self.attrs_2_building is None:
            self.attrs_2_building = {}
        for idx,(key,_) in enumerate(dd.items()):
            tmp_key_sort = tuple(sorted(key))
            if tmp_key_sort not in self.attrs_2_building.keys():
                if _ < connect_threshold: # 两个建筑物之间的三角形少于2个，不做考虑
                    continue
                re = self.calculate_next2_features(key[0],key[1])
                if re == -1:
                    continue
                self.attrs_2_building[tuple(sorted(key))] = re
        # print(f'TIME:{time.ctime()}----> END CALCULATE RELATIONSHIP ATTRS')

    def my_over_lap(self,building_idx1,building_idx2):
        b1 = self.get_rectangle_angle_and_length(geometry.Polygon(self.shapes[building_idx1].points).minimum_rotated_rectangle)
        b2 = self.get_rectangle_angle_and_length(geometry.Polygon(self.shapes[building_idx2].points).minimum_rotated_rectangle)
        angle_cos_abs = np.abs(np.cos(b1['angle_radians'] - b2['angle_radians']))
        proj_1_2 = b1['long_length'] * angle_cos_abs
        proj_2_1 = b2['long_length'] * angle_cos_abs
        return ((min(b1['long_length'], proj_2_1) / max(b1['long_length'], proj_2_1))+(min(b2['long_length'], proj_1_2) / max(b2['long_length'], proj_1_2)))/2.

    def global_features(self):
        """
        建筑群特征
        """
        if self.global_attrs is None:
            self.global_attrs = {}
        recty = self.buildings_rectangularity(self.shapes)
        self.global_attrs['rectangularity'] = recty

    def calculate_triangle_3_angles(self,triangle_vertices):
        """
        计算三角形的三个角度
        """
        a = np.linalg.norm(triangle_vertices[:,1,:] - triangle_vertices[:,2,:],axis=1)
        b = np.linalg.norm(triangle_vertices[:,0,:] - triangle_vertices[:,2,:],axis=1)
        c = np.linalg.norm(triangle_vertices[:,0,:] - triangle_vertices[:,1,:],axis=1)
        cos_a = (b**2 + c**2 - a**2) / ((2 * b * c)+1e-8)
        cos_b = (a**2 + c**2 - b**2) / ((2 * a * c)+1e-8)
        cos_c = (a**2 + b**2 - c**2) / ((2 * a * b)+1e-8)
        degree_a = np.degrees(np.arccos(cos_a))
        degree_b = np.degrees(np.arccos(cos_b))
        degree_c = np.degrees(np.arccos(cos_c))
        # return np.arccos(cos_a),np.arccos(cos_b),np.arccos(cos_c)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        area = np.where(np.isnan(area),0,area)
        return np.asarray([degree_a,degree_b,degree_c]).T, np.asarray([2 * area / a, 2 * area /b, 2 * area / c]).T, np.asarray([a,b,c]).T

    def visual_distance(self,building_idx1,building_idx2) -> float:
        """
        可视距离
        """
        # 找到两个建筑物之间的骨架线的索引
        skeleton_ids = list(set(self.skeleton.polygon_2_skeleton[building_idx1])&set(self.skeleton.polygon_2_skeleton[building_idx2]))
        # 删除与建筑物相交的骨架线idx，不做考虑
        overlap_buildings_lines = self.skeleton.overlap_buildings_lines
        if len(overlap_buildings_lines) > 0 :
            need_del_idx = []
            for del_idx in self.skeleton.overlap_buildings_lines:
                r = np.argwhere(np.asarray(skeleton_ids) == del_idx)
                if r.shape[0] != 0:
                    need_del_idx.append(r[0,0])
            if len(need_del_idx) > 0:
                skeleton_ids = np.delete(skeleton_ids,need_del_idx,axis=0)
        triangle_ids = np.asarray(self.skeleton.skeleton_from_triangle)[skeleton_ids]
        triangle_vertices_from_polygon = self.tri['triangle_node_from_polygon'][triangle_ids]
        # 只考虑两个建筑物之间的三角形
        involved_triangle_ids = np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=triangle_vertices_from_polygon) # 每行元素的种类数量
        skeleton_ids = np.asarray(skeleton_ids)[np.where(involved_triangle_ids == 2)[0]]
        if len(overlap_buildings_lines) > 0 :
            need_del_idx = []
            for del_idx in self.skeleton.overlap_buildings_lines:
                r = np.argwhere(np.asarray(skeleton_ids) == del_idx)
                if r.shape[0] != 0:
                    need_del_idx.append(r[0,0])
            if len(need_del_idx) > 0:
                skeleton_ids = np.delete(skeleton_ids,need_del_idx,axis=0)
        triangle_ids = np.asarray(self.skeleton.skeleton_from_triangle)[skeleton_ids]
        triangle_vertices_from_polygon = self.tri['triangle_node_from_polygon'][triangle_ids]
        triangle_vertices = self.tri['triangle_vertices'][triangle_ids]
        angles, heights,edge_lens = self.calculate_triangle_3_angles(triangle_vertices)
        # 获取每行中相同元素的列索引
        same_element_indices = np.asarray(list(map(lambda row: np.where(row == np.unique(row, return_counts=True)[0][np.unique(row, return_counts=True)[1] > 1][:, np.newaxis])[1], triangle_vertices_from_polygon))) # 获取在同一三角形上的两个点的索引
        # 获取剩下的一个元素的列索引
        rest_element_indices = np.asarray(list(map(lambda row: np.where(row != np.unique(row, return_counts=True)[0][np.unique(row, return_counts=True)[1] > 1][:, np.newaxis])[1], triangle_vertices_from_polygon)))
        # 判断同一个建筑物上两个角是不是都是锐角，要是有钝角，找出哪个是钝角
        angle_on_same_building = angles[np.arange(same_element_indices.shape[0])[:,None],same_element_indices]
        is_obtuse_angle = np.where(angle_on_same_building > 90,1,0)
        is_obtuse_angle_col_exchange = is_obtuse_angle[:,[1,0]] # 交换之后，如果是钝角，直接取对应索引的边作为高度，不用另一个角的对应边
        visual_dist = 0
        length_skeleton = np.asarray([self.get_angle_and_length(self.skeleton.lines[i])['length'] for i in skeleton_ids])
        for iter, col in enumerate(is_obtuse_angle_col_exchange):
            if not 1 in col:# 锐角三角形
                visual_dist += heights[iter,rest_element_indices[iter][0]] * length_skeleton[iter]
            else:
                visual_dist += edge_lens[iter,np.where(col==1)[0][0]] * length_skeleton[iter]
        visual_dist /= length_skeleton.sum()
        return visual_dist

    def calculate_next2_features(self,building_idx1,building_idx2,common_edge=False):
        centroid1 = self.attrs[building_idx1]['centroid']
        centroid2 = self.attrs[building_idx2]['centroid']
        angle_and_length = self.get_angle_and_length([centroid1,centroid2])
        Fr = self.my_over_lap(building_idx1,building_idx2)# self.calculate_FR_wei(building_idx1,building_idx2)
        Dr_angle = angle_and_length['angle_radians']
        o_dist = angle_and_length['length']
        rectangularity = self.buildings_rectangularity([self.shapes[building_idx1],self.shapes[building_idx2]])
        # v_dist = None
        if not common_edge:
            try:
                v_dist = self.visual_distance(building_idx1,building_idx2)
            except:
                return -1
        else:
            v_dist = 0.0000001
        return {
            'Fr':Fr,
            'Dr_angle':Dr_angle,
            'o_dist':o_dist,
            'rectangularity':rectangularity,
            'v_dist':v_dist
        }

    def point_2_line_projection(self,point,line):
        """
        点到线的投影点
        """
        x0,y0 = point
        x1,y1 = line[0]
        x2,y2 = line[1]
        if x1 == x2:
            return x1,y0
        if y1 == y2:
            return x0,y1
        k = (y2-y1)/(x2-x1)
        b = y1 - k*x1
        x = (k*y0+x0-k*b)/(k*k+1)
        y = k*x+b
        return [x,y]

    def create_big_hull(self):
        """
        建筑群凸包外扩
        """
        if self.shapes is None:
            self.shapes = self.shp_file.shapes()
        big_hull = geometry.MultiPolygon([geometry.Polygon(i.points) for i in self.shapes]).convex_hull
        # big_rectangle = big_hull.minimum_rotated_rectangle
        # 确定外扩距离,随机选择一个多边形数据,以多边形外接矩形短边的一半作为外扩距离
        # big_rect_length = self.get_rectangle_angle_and_length(big_rectangle)['long_length'] + self.get_rectangle_angle_and_length(big_rectangle)['short_length']
        building_num = len(self.shapes)
        #开根号
        # building_num_sqrt = math.sqrt(building_num)
        extend = 2.5 # big_rect_length/(building_num_sqrt*20)
        return list(map(list, geometry.mapping(geometry.Polygon(big_hull).buffer(extend*5,join_style='mitre'))['coordinates'][0])),extend

    def create_skeleton(self):
        if self.skeleton is None:
            self.skeleton = Skeleton()
        self.skeleton.load_triangle_data(self.tri)
        self.skeleton.analysis_triangle_data()
        self.skeleton.delete_lines_overlap_buildings(self.tree)
        # self.skeleton.lines_2_polys()

    def tmp_save_lines_to_shp(self):
        # 保存三角网数据和骨架线数据
        # print(f'TIME:{time.ctime()}----> START SAVE LINES TO SHP')
        w = shapefile.Writer('/to/path/CODE/predata/outs/s_lines1')
        w.field('id', 'C')
        ske_info = self.skeleton.get_result()
        for idx,line in enumerate(ske_info['skeletons']):
            line = line.tolist()
            # if idx in ske_info['polygon_2_skeleton'][-2]:
            #     w.line([line])
            #     w.record(str(idx))
            w.line([line])
            w.record(str(idx))
        w.close()
        # print(f'TIME:{time.ctime()}----> END SAVE LINES TO SHP')
        # print(f'TIME:{time.ctime()}----> START SAVE TRIANGLES TO SHP')
        w = shapefile.Writer('/to/path/CODE/predata/outs/triangles')
        w.field('id', 'C')
        triangle_vertices = self.tri['triangle_vertices']
        # del_index = self.tri['del_index']
        # triangle_vertices = np.delete(triangle_vertices,del_index,axis=0)
        for idx,triangle in enumerate(triangle_vertices):
            triangle = triangle.tolist()
            w.poly([triangle])
            w.record(str(idx))
        w.close()
        # print(f'TIME:{time.ctime()}----> END SAVE TRIANGLES TO SHP')
        # 保存点
        # print(f'TIME:{time.ctime()}----> START SAVE POINTS TO SHP')
        w = shapefile.Writer('/to/path/CODE/predata/outs/points')
        w.field('id', 'C')
        points = self.extend_points
        for idx,point in enumerate(points):
            w.point(*point)
            w.record(str(idx))
        w.close()

    def get_building_data(self):
        return self.shapes,self.attrs_2_building

    def create_pictures(self):
        delaunay = np.delete(self.tri['tri_s'].simplices,self.tri['del_index'],axis=0)
        points = np.asarray(self.extend_points)
        name = self.building_type + '_' + self.file_name.split('/')[-1]
        self.name = name
        if self.png_path is not None:
            file_dir = self.png_path
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path = os.path.join(file_dir, f"{name}.svg")
            # plt.triplot(points[:, 0], points[:, 1], delaunay, linewidth=.1)
            for i in self.shapes:
                plt.plot(*geometry.Polygon(i.points).exterior.xy)
                # 画建筑物centroid点
                plt.plot(*geometry.Polygon(i.points).centroid.xy, 'o',color='black')
            for i in self.attrs_2_building.keys():
                # centroid line
                centroid1 = self.attrs[i[0]]['centroid']
                centroid2 = self.attrs[i[1]]['centroid']
                # 画线
                plt.plot([centroid1[0],centroid2[0]],[centroid1[1],centroid2[1]],color='red')
            plt.savefig(file_path)
            plt.close()
    
    def other_norm(self, x):
        x = np.asarray(x)
        x = x - np.mean(x)
        # x = abs(x)
        return x.tolist()

    def log_norm(self, x):
        x = np.asarray(x)
        x = np.log1p(x)
        x = x - np.mean(x)
        # x = abs(x)
        return x.tolist()

    def angle_norm(self, x, max_angle=math.pi):
        x = np.asarray(x)
        x = x - np.mean(x)
        x = x / max_angle
        # x = abs(x)
        return x.tolist()
    
    def raito_norm(self, x):
        x = np.asarray(x)
        x = x - np.mean(x)
        # x = abs(x)
        return x.tolist()

    def get_result(self):
        self.global_features()
        global_features = self.global_attrs
        name = self.name
        graph_index = [[i[0],i[1],iter] for iter,i in enumerate(self.attrs_2_building.keys())]
        graph_index = np.int8(graph_index).tolist()
        e_features = np.asarray([[i[j]  for j in i.keys()] for i in self.attrs_2_building.values()])
        n_features = np.asarray([[i[j] for j in i.keys() if j != 'centroid'] for i in self.attrs])

        e_features[:,0] = np.asarray(self.raito_norm(e_features[:,0]))
        e_features[:,1] = np.asarray(self.angle_norm(e_features[:,1]))
        e_features[:,2] = np.asarray(self.other_norm(e_features[:,2]))
        e_features[:,3] = np.asarray(self.raito_norm(e_features[:,3]))
        e_features[:,4] = np.asarray(self.other_norm(e_features[:,4]))

        n_features = np.append(n_features,np.array([global_features['rectangularity']] * n_features.shape[0]).reshape(-1,1),axis=1)

        n_features[:,0] = np.asarray(self.other_norm(n_features[:,0]))
        n_features[:,1] = np.asarray(self.angle_norm(n_features[:,1]))
        n_features[:,2] = np.asarray(self.raito_norm(n_features[:,2]))
        n_features[:,3] = np.asarray(self.raito_norm(n_features[:,3]))
        n_features[:,4] = np.asarray(self.log_norm(n_features[:,4]))

        n_features = n_features.tolist()
        e_features = e_features.tolist()
        return {
            "graph_index":graph_index,
            "file_path":name,
            "datas":{
                "node_features":n_features,
                "edge_features":e_features,
                "label":self.building_label,
                "points":[i.points for i in self.shapes]
            }
        }



