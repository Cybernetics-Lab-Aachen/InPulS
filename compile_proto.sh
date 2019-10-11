PROTO_SRC_DIR=proto
DST_DIR=build
# Hack to compile directly into src folders for now
PROTO_BUILD_DIR=$DST_DIR/$PROTO_SRC_DIR
PY_PROTO_BUILD_DIR=gps/proto

mkdir -p "$PROTO_BUILD_DIR"
mkdir -p "$PY_PROTO_BUILD_DIR"
touch $PY_PROTO_BUILD_DIR/__init__.py

protoc -I=$PROTO_SRC_DIR --python_out=$PY_PROTO_BUILD_DIR $PROTO_SRC_DIR/gps.proto $PROTO_SRC_DIR/Command.proto

echo "Done"
