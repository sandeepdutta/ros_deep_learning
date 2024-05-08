from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # Declare arguments
    model_name_arg = DeclareLaunchArgument(
        'model_name', default_value='ssd-mobilenet-v2')
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='')
    prototxt_path_arg = DeclareLaunchArgument(
        'prototxt_path', default_value='')
    class_labels_path_arg = DeclareLaunchArgument(
        'class_labels_path', default_value='')
    input_blob_arg = DeclareLaunchArgument(
        'input_blob', default_value='')
    output_cvg_arg = DeclareLaunchArgument(
        'output_cvg', default_value='')
    output_bbox_arg = DeclareLaunchArgument(
        'output_bbox', default_value='')
    overlay_flags_arg = DeclareLaunchArgument(
        'overlay_flags', default_value='none')#'box,labels,conf')
    mean_pixel_value_arg = DeclareLaunchArgument(
        'mean_pixel_value', default_value='0.0')
    threshold_arg = DeclareLaunchArgument(
        'threshold', default_value='0.5')

    # Include video source launch file
    video_source_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                ThisLaunchFileDir(), 'video_source.ros2.launch.py'
            ])
        ])
    )

    # Node for detectnet
    detectnet_node = Node(
        package='ros_deep_learning',
        executable='detectnet',
        output='screen',
        remappings=[('image_in_color', '/rs_top/camera/color/image_raw'),
                    ('image_in_depth', '/rs_top/camera/depth/image_rect_raw')],
        parameters=[{
            'model_name': LaunchConfiguration('model_name'),
            'model_path': LaunchConfiguration('model_path'),
            'prototxt_path': LaunchConfiguration('prototxt_path'),
            'class_labels_path': LaunchConfiguration('class_labels_path'),
            'input_blob': LaunchConfiguration('input_blob'),
            'output_cvg': LaunchConfiguration('output_cvg'),
            'output_bbox': LaunchConfiguration('output_bbox'),
            'overlay_flags': LaunchConfiguration('overlay_flags'),
            'mean_pixel_value': LaunchConfiguration('mean_pixel_value'),
            'threshold': LaunchConfiguration('threshold')
        }]
    )

    # Include video output launch file
    video_output_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                ThisLaunchFileDir(), 'video_output.ros2.launch.py'
            ])
        ]),
        launch_arguments={'topic': '/detectnet/overlay'}.items()
    )

    # Assemble the launch description
    return LaunchDescription([
        model_name_arg,
        model_path_arg,
        prototxt_path_arg,
        class_labels_path_arg,
        input_blob_arg,
        output_cvg_arg,
        output_bbox_arg,
        overlay_flags_arg,
        mean_pixel_value_arg,
        threshold_arg,
       #video_source_launch,
       #video_output_launch,
        detectnet_node
    ])
