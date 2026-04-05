//==============================================================================
// Copyright (c) 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================
#include <gtest/gtest.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/utils/EnvUtils.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::utils::GetEnv;
using setu::commons::utils::ParseValue;
//==============================================================================

// --- ParseValue tests --------------------------------------------------------

TEST(ParseValueTest, SizeT) {
  EXPECT_EQ(ParseValue<std::size_t>("42"), 42);
  EXPECT_EQ(ParseValue<std::size_t>("0"), 0);
}

TEST(ParseValueTest, Int32) {
  EXPECT_EQ(ParseValue<std::int32_t>("123"), 123);
  EXPECT_EQ(ParseValue<std::int32_t>("-5"), -5);
}

TEST(ParseValueTest, Int64) {
  EXPECT_EQ(ParseValue<std::int64_t>("9999999999"), 9999999999LL);
  EXPECT_EQ(ParseValue<std::int64_t>("-1"), -1);
}

TEST(ParseValueTest, Bool) {
  EXPECT_TRUE(ParseValue<bool>("1"));
  EXPECT_TRUE(ParseValue<bool>("true"));
  EXPECT_TRUE(ParseValue<bool>("TRUE"));
  EXPECT_FALSE(ParseValue<bool>("0"));
  EXPECT_FALSE(ParseValue<bool>("false"));
  EXPECT_FALSE(ParseValue<bool>("anything"));
}

TEST(ParseValueTest, String) {
  EXPECT_EQ(ParseValue<std::string>("hello"), "hello");
  EXPECT_EQ(ParseValue<std::string>(""), "");
}

// --- GetEnv scalar tests -----------------------------------------------------

class GetEnvTest : public ::testing::Test {
 protected:
  void SetEnv(const char* name, const char* value) {
    setenv(name, value, 1);
    env_vars_.push_back(name);
  }

  void TearDown() override {
    for (const auto& name : env_vars_) {
      unsetenv(name.c_str());
    }
  }

 private:
  std::vector<std::string> env_vars_;
};

TEST_F(GetEnvTest, ReturnsDefault_WhenUnset) {
  EXPECT_EQ(GetEnv<std::size_t>("SETU_TEST_UNSET_VAR", 99), 99);
  EXPECT_EQ(GetEnv<bool>("SETU_TEST_UNSET_VAR", true), true);
  EXPECT_EQ(GetEnv<std::string>("SETU_TEST_UNSET_VAR", "default"), "default");
}

TEST_F(GetEnvTest, SizeT) {
  SetEnv("SETU_TEST_SIZE", "256");
  EXPECT_EQ(GetEnv<std::size_t>("SETU_TEST_SIZE", 0), 256);
}

TEST_F(GetEnvTest, Int32) {
  SetEnv("SETU_TEST_INT", "-10");
  EXPECT_EQ(GetEnv<std::int32_t>("SETU_TEST_INT", 0), -10);
}

TEST_F(GetEnvTest, Bool_True) {
  SetEnv("SETU_TEST_BOOL", "true");
  EXPECT_TRUE(GetEnv<bool>("SETU_TEST_BOOL", false));
}

TEST_F(GetEnvTest, Bool_False) {
  SetEnv("SETU_TEST_BOOL", "0");
  EXPECT_FALSE(GetEnv<bool>("SETU_TEST_BOOL", true));
}

TEST_F(GetEnvTest, String) {
  SetEnv("SETU_TEST_STR", "hello world");
  EXPECT_EQ(GetEnv<std::string>("SETU_TEST_STR", ""), "hello world");
}

// --- GetEnv vector tests -----------------------------------------------------

TEST_F(GetEnvTest, Vector_ReturnsDefault_WhenUnset) {
  std::vector<std::int32_t> def = {1, 2, 3};
  EXPECT_EQ(GetEnv<std::int32_t>("SETU_TEST_UNSET_VEC", def), def);
}

TEST_F(GetEnvTest, Vector_Int) {
  SetEnv("SETU_TEST_VEC_INT", "10,20,30");
  std::vector<std::int32_t> expected = {10, 20, 30};
  std::vector<std::int32_t> empty = {};
  EXPECT_EQ(GetEnv("SETU_TEST_VEC_INT", empty), expected);
}

TEST_F(GetEnvTest, Vector_SizeT) {
  SetEnv("SETU_TEST_VEC_SIZE", "1,2,3,4");
  std::vector<std::size_t> expected = {1, 2, 3, 4};
  std::vector<std::size_t> empty = {};
  EXPECT_EQ(GetEnv("SETU_TEST_VEC_SIZE", empty), expected);
}

TEST_F(GetEnvTest, Vector_String) {
  SetEnv("SETU_TEST_VEC_STR", "foo,bar,baz");
  std::vector<std::string> expected = {"foo", "bar", "baz"};
  std::vector<std::string> empty = {};
  EXPECT_EQ(GetEnv("SETU_TEST_VEC_STR", empty), expected);
}

TEST_F(GetEnvTest, Vector_SingleElement) {
  SetEnv("SETU_TEST_VEC_ONE", "42");
  std::vector<std::int32_t> expected = {42};
  std::vector<std::int32_t> empty = {};
  EXPECT_EQ(GetEnv("SETU_TEST_VEC_ONE", empty), expected);
}

TEST_F(GetEnvTest, Vector_Bool) {
  SetEnv("SETU_TEST_VEC_BOOL", "1,false,TRUE,0");
  std::vector<bool> expected = {true, false, true, false};
  std::vector<bool> empty = {};
  EXPECT_EQ(GetEnv("SETU_TEST_VEC_BOOL", empty), expected);
}

//==============================================================================
}  // namespace setu::test::native
//==============================================================================
