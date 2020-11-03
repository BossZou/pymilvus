import pytest

from milvus import DataType, BaseError, ParamError

from factorys import collection_name_factory


class TestGetCollectionInfo:
    def test_get_collection_info_normal(self, connect, vrecords):
        try:
            info = connect.get_collection_info(vrecords)
            assert info["auto_id"]
            for field in info["fields"]:
                if field["name"] == "Int":
                    assert field["type"] == DataType.INT64
                if field["name"] == "Vec":
                    assert field["type"] == DataType.FLOAT_VECTOR
        except Exception as e:
            pytest.fail(f"Get collection {vrecords} info fail: {str(e)}")

    def test_get_collection_info_not_exist(self, connect):
        with pytest.raises(BaseError):
            collection_name = collection_name_factory()
            _ = connect.get_collection_info(collection_name)

    @pytest.mark.parametrize("name", ["", 124, [1]])
    def test_get_collection_info_invalid_name(self, name, connect):
        with pytest.raises(ParamError):
            _ = connect.get_collection_info(name)
