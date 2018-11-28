PROTO_SRC_DIR=src/proto
DST_DIR=build
# Hack to compile directly into src folders for now
CPP_OUT_DIR=src/gps_agent_pkg/include/gps/proto
PROTO_BUILD_DIR=$DST_DIR/$PROTO_SRC_DIR
PY_PROTO_BUILD_DIR=python/gps/proto

#former ROS message -> now proto

MESSAGE_PROTO_SRC_DIR=src/gps_agent_pkg/proto
MESSAGE_CPP_OUT_DIR=src/gps_agent_pkg/proto/cpp_compiled
MESSAGE_PY_OUT_DIR=src/gps_agent_pkg/proto/python_compiled


mkdir -p "$PROTO_BUILD_DIR"
mkdir -p "$PY_PROTO_BUILD_DIR"
touch $PY_PROTO_BUILD_DIR/__init__.py

mkdir -p "$CPP_OUT_DIR"
protoc -I=$PROTO_SRC_DIR --cpp_out=$CPP_OUT_DIR $PROTO_SRC_DIR/gps.proto
protoc -I=$PROTO_SRC_DIR --python_out=$PY_PROTO_BUILD_DIR $PROTO_SRC_DIR/gps.proto

mkdir -p "$MESSAGE_CPP_OUT_DIR"
mkdir -p "$MESSAGE_PY_OUT_DIR"
touch $MESSAGE_PY_OUT_DIR/__init__.py

protoc -I=$MESSAGE_PROTO_SRC_DIR --cpp_out=$MESSAGE_CPP_OUT_DIR --python_out=$MESSAGE_PY_OUT_DIR $MESSAGE_PROTO_SRC_DIR/*.proto

echo "Done"
