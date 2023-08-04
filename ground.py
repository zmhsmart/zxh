import torch
import smplx
import math
import numpy as np
import time
import trimesh
from sklearn.cluster import KMeans


def plane(vetices,variable, norml):
    
    """
    利用地平面的法向量和膝关节点，利用点法式方程确立出过膝关节且平行与地面的平面
    vetices 判断是否在平面上，平面外任意一点，用来判断该点是否在拟合的地平面
    variable   拟合的平面上的点，膝关节上的点
    norml    地面计算法向量，地面的三个点用来确定平面法向量
    """
    x1 = norml[0][0]
    y1 = norml[0][1]
    z1 = norml[0][2]
    x2 = norml[1][0]
    y2 = norml[1][1]
    z2 = norml[1][2]
    x3 = norml[2][0]
    y3 = norml[2][1]
    z3 = norml[2][2]
    A = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1)
    B = (x3-x1)*(z2-z1) - (x2-x1)*(z3-z1)
    C = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    # # 单位化
    # length = math.sqrt(A**2 + B**2 + C**2)
    # A1 = A / length
    # B1 = B / length
    # C1 = C / length

    # 点法式方程
    D = -(A*variable[0] + B*variable[1] + C*variable[2])
    # print('平面合结果拟为:%.3f * x + %.3f * y + %.3f *z + %.3f = 0' % (A, B, C, D))
    # 判断点和平面关系
    ans = A*vetices[0] + B*vetices[1] + C*vetices[2] + D

    return ans


def plane2(variable, norml):

    """
    利用地平面的法向量和膝关节点，利用点法式方程确立出过膝关节且平行与地面的平面
    vetices 判断是否在平面上，平面外任意一点，用来判断该点是否在拟合的地平面
    variable   拟合的平面上的点，膝关节上的点
    norml    地面计算法向量，地面的三个点用来确定平面法向量
    """
    x1 = norml[0][0]
    y1 = norml[0][1]
    z1 = norml[0][2]
    x2 = norml[1][0]
    y2 = norml[1][1]
    z2 = norml[1][2]
    x3 = norml[2][0]
    y3 = norml[2][1]
    z3 = norml[2][2]
    A = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1)
    B = (x3-x1)*(z2-z1) - (x2-x1)*(z3-z1)
    C = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    # # 单位化
    # length = math.sqrt(A**2 + B**2 + C**2)
    # A1 = A / length
    # B1 = B / length
    # C1 = C / length

    # 点法式方程
    D = -(A*variable[0] + B*variable[1] + C*variable[2])
    print('平面合结果拟为:%.3f * x + %.3f * y + %.3f *z + %.3f = 0' % (A, B, C, D))
    # 判断点和平面关系
    

    return B,D    



def cross_points(point_1, point_2, point):
    '''
    判断三角面片与与平面的交点
    '''
    x1 = point[0][0]
    y1 = point[0][1]
    z1 = point[0][2]
    x2 = point[1][0]
    y2 = point[1][1]
    z2 = point[1][2]
    x3 = point[2][0]
    y3 = point[2][1]
    z3 = point[2][2]
    A = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1)
    B = (x3-x1)*(z2-z1) - (x2-x1)*(z3-z1)
    C = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    m1 = point_1[0]
    m2 = point_1[1]
    m3 = point_1[2]
    v1 = point_2[0] - point_1[0]
    v2 = point_2[1] - point_1[1]
    v3 = point_2[2] - point_1[2]
    vp1 = A
    vp2 = B
    vp3 = C
    n1 = x2
    n2 = y2
    n3 = z2
    vpt = v1 * vp1 + v2 * vp2 + v3 * vp3
    t = ((n1 - m1) * vp1 + (n2 - m2) * vp2 + (n3 - m3) * vp3) / vpt
    x = m1 + v1 * t
    y = m2 + v2 * t
    z = m3 + v3 * t
    point_t = point_1 * 1


    point_t[0] = x
    point_t[1] = y
    point_t[2] = z


    return point_t




# 读取数据
with open('/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/shape/180_betas.txt', 'r') as f:
    beta_values = np.loadtxt(f)
with open('/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/pose/180_pose.txt', 'r') as f:
    pose_values = np.loadtxt(f)

body_model = smplx.create(model_path="SMPLX_MALE.npz", model_type='smplx')

with torch.no_grad():
    body_model.betas[0] = torch.tensor(beta_values)
    body_model.body_pose[0] = torch.tensor(pose_values)
body_model_output= body_model(return_verts=True, return_full_pose=False)
faces = body_model.faces                # (20908,3)
verts = body_model_output.vertices
verts = verts.squeeze(0)   
joints = body_model_output.joints 


out_mesh = trimesh.Trimesh(verts.detach().cpu().numpy().squeeze(), body_model.faces, process=False)
out_mesh.export("/home/pbc/project/zxh/body shape measurement/180/180.obj")



# 地平面的三个坐标点，用来拟合平面
ground_idx = np.loadtxt('/home/pbc/project/inference/pose_3d/ground_point_in_smpl.txt')
delta= np.loadtxt("/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/delta.txt")
ground_idx[:,0]-=delta[0]
ground_idx[:,1]-=delta[1]
ground_idx[:,2]-=delta[2]
plane_points = ground_idx[[0,5,9]]


# 膝盖的关节点，点法式确立方程的点
variable = [0, -5/6, 0]
# variable = joints[0,5]
# variable = verts[6400]

# 脚踝的关节点，点法式确立方程的点
# variable_ankle = [0,-7/6 , 0]
# variable_ankle = joints[0,8]
variable_ankle = verts[8447]


# pathlegl = "/home/pbc/project/inference/func_regress/partid/legl_idx.txt"  
pathlegl = "/home/pbc/project/inference/func_regress/partid/legr_idx.txt"  
# pathlegl = "/home/pbc/project/inference/func_regress/partid/126_knee_idx.txt"  
# pathlegl = "/home/pbc/project/inference/func_regress/partid/leg1213_idx.txt"
# pathlegl = "/home/pbc/project/inference/func_regress/partid/leg_idx.txt"     
with open(pathlegl,'r') as f:
        body_idx = f.readlines()   


# # #L
# point_1 = [0, -5/6, 0]
# point_2 = [1, -5/6, 1]  
# point_3 = [1, -5/6, -1]

# 膝盖
B,D = plane2(variable,plane_points)

point_1 = [0, -D/B, 0]
point_2 = [1, -D/B, 1] 
point_3 = [1, -D/B, -1]

# 脚踝
B,D = plane2(variable_ankle,plane_points)


point_4 = [0, -D/B, 0]
point_5 = [1, -D/B, 1]  
point_6 = [1, -D/B, -1]
# point_4 = joints[0,8]
# point_5 = joints[0,7]
# point_6 = verts[5753]
# point_4 = [0, -8/6, 0]
# point_5 = [1, -8/6, 1]  
# point_6 = [1, -8/6, -1]

point_kneel = [point_1, point_2, point_3]

point_Anklel = [point_4, point_5, point_6]

# 膝关节平面选取的三个点
# point_kneel = []
# point_kneel.append(joints[0,5])
# point_kneel.append(joints[0,4])
# point_kneel.append(verts[3781])
# point_kneel.append(verts[3683])
# point_kneel.append(verts[3650])
# point_kneel.append(verts[3781])



# point_Anklel = []
# point_Anklel.append(joints[0,8])
# point_Anklel.append(joints[0,7])
# point_Anklel.append(verts[5878])
# point_Anklel.append(verts[5880])
# point_Anklel.append(verts[5753])
# point_Anklel.append(verts[5878])
start_time = time.time()


mesh_points_knee=[]

mesh_points_ankle=[]
len_total = 0
for i in range(len(faces)):
        
        n =0
        p1_idx = faces[i,0]
        p2_idx = faces[i,1]
        p3_idx = faces[i,2]
        # print(str(p1_idx))
        if ((str(p1_idx)+'\n') not in body_idx) and ((str(p2_idx)+'\n') not in body_idx) and ((str(p3_idx)+'\n') not in body_idx) :          # 判断相交mesh是否在所需部位
            continue
        # print(str(p1_idx))
        point1 = verts[p1_idx,:]
        point2 = verts[p2_idx,:]
        point3 = verts[p3_idx,:]


        # 说明三个点都在平面同一侧，三角面片和地平面不相交，所以没有交点,应该让points1，2，3  
        
        ans1= plane(point1,variable,plane_points)
        ans2= plane(point2,variable,plane_points)
        ans3= plane(point3,variable,plane_points)

        # 脚踝
        ans4= plane(point1,variable_ankle,plane_points)
        ans5= plane(point2,variable_ankle,plane_points)
        ans6= plane(point3,variable_ankle,plane_points)
        # print(ans1,ans2,ans3)
        # 膝盖判定
        mask1 = (ans1 * ans2 <= 0)
        mask2 = (ans2 * ans3 <= 0)
        mask3 = (ans1 * ans3 <= 0)

        # 脚踝判定
        mask4 = (ans4 * ans5 <= 0)
        mask5 = (ans5 * ans6 <= 0)
        mask6 = (ans4 * ans6 <= 0)
        # 膝盖
        if mask1:
            point_1 = cross_points(point1, point2, point_kneel)
            mesh_points_knee.append(point_1)
            n += 1
        if mask2:
            point_2 = cross_points(point2, point3, point_kneel)
            mesh_points_knee.append(point_2)
            n += 1
        if mask3:
            point_3 = cross_points(point1, point3, point_kneel)
            mesh_points_knee.append(point_3)
            n += 1
        
        
        # 脚踝交点
        if mask4:
            point_1 = cross_points(point1, point2, point_Anklel)
            mesh_points_ankle.append(point_1)
            n += 1
        if mask5:
            point_2= cross_points(point2, point3, point_Anklel)
            mesh_points_ankle.append(point_2)
            n += 1
        if mask6:
            point_3 = cross_points(point1, point3, point_Anklel)
            mesh_points_ankle.append(point_3)
            n += 1
    
        # 围度测量
        

mesh_points_knee = np.array([tensor.detach().numpy() for tensor in mesh_points_knee],dtype=float)
mesh_points_ankle = np.array([tensor.detach().numpy() for tensor in mesh_points_ankle],dtype=float)




# print(mesh_points)
# 保存为文本文件
np.savetxt('body shape measurement/180/mesh_points_knee_180_r .txt', mesh_points_knee)
np.savetxt('body shape measurement/180/mesh_points_ankle_180_r .txt', mesh_points_ankle)
# np.savetxt('body shape measurement/test/mesh_points_ankle_126_l .txt', dense_points)



# 打印保存的文件内容
print("文件已保存：")
end_time = time.time()
all_time = end_time-start_time
print('所需时间为:',all_time)
# with open('mesh_points.txt', 'r') as f:
#     print(f.read())
