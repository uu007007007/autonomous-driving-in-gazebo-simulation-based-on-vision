
from ...config import config

# ****************************************************  DETECTION ****************************************************
# ****************************************************    LANES   ****************************************************

# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 1 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from .a_Segmentation.colour_segmentation_final import Segment_Colour

# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 2 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 이상적인 중간 차선 추정
from .b_Estimation.Our_EstimationAlgo import Estimate_MidLane
# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 3 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 차선 경계선 정리 및 확장
from .c_Cleaning.CheckifYellowLaneCorrect_RetInnerBoundary import GetYellowInnerEdge
from .c_Cleaning.ExtendLanesAndRefineMidLaneEdge import ExtendShortLane
# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 4 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 유효한 정보를 추출하고 차량 위치 추정
from .d_LaneInfo_Extraction.GetStateInfoandDisplayLane import FetchInfoAndDisplay


def detect_Lane(img):
        """ Extract required data from the lane lines representing road lane boundaries.

        Args:
                img (numpy nd array): Prius front-cam view

        Returns:
                distance    (int): car_front <===distance===> ideal position on road 
                curvature (angle): car <===angle===> roads_direction
                                e.g. car approaching a right turn so road direction is around or less then 45 deg
                                                                                cars direction is straight so it is around 90 deg
        """          
        # >>>>>>>>>>>>>>>>>>>>>>>> Optimization No 2 [CROPPING] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # 상단 부분을 잘라내고 하단 부분만 사용, 차선 정보만 분석
        img_cropped = img[config.CropHeight_resized:,:]

        # [Lane Detection] STAGE_1 (Segmentation) <<<<<<--->>>>>> [COLOR]:
        # 중간 및 외부 차선 엣지와 마스크 추출
        Mid_edge_ROI,Mid_ROI_mask,Outer_edge_ROI,OuterLane_TwoSide,OuterLane_Points = Segment_Colour(img_cropped,config.minArea_resized)
       
        # [Lane Detection] STAGE_2 (Estimation) <<<<<<--->>>>>> [Our Approach]:
        # 중간 차선 위치 추정, 나머지 차선과 상대적 거리 계산
        Estimated_midlane = Estimate_MidLane(Mid_edge_ROI,config.MaxDist_resized)

        # [Lane Detection] STAGE_3 (Cleaning) <<<<<<--->>>>>> [STEP_1]:
        # 경계 파악
        OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction = GetYellowInnerEdge(OuterLane_TwoSide,Estimated_midlane,OuterLane_Points)#3ms
        # [Lane Detection] STAGE_3 (Cleaning) <<<<<<--->>>>>> [STEP_2]:
        # 차선이 짧은 경우 확장
        Estimated_midlane,OuterLane_OneSide = ExtendShortLane(Estimated_midlane,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)
        
        # [Lane Detection] STAGE_4 (Data_Extraction) <<<<<<--->>>>>> [Our Approach]:
        # 도로에서 위치과 곡률 계산
        Distance , Curvature = FetchInfoAndDisplay(Mid_edge_ROI,Estimated_midlane,OuterLane_OneSide,img_cropped,Offset_correction)

        return Distance, Curvature
       