#include <iostream>
#include <string>
#include <vector>
#include "unit_test/UnitTest++.h"
#include "rrc_api.h"
#include "rrUtils.h"
#include "rrException.h"
#include "src/rrSBMLTestSuiteSimulation_CAPI.h"

using namespace UnitTest;
using namespace std;
using namespace rr;

extern string     gTSModelsPath;
extern string   gTempFolder;
extern bool        gDebug;

SUITE(SBML_TEST_SUITE)
{
 /*TEST(1) {CHECK(RunTest("l2v4", 1)); }
 TEST(2) { CHECK(RunTest("l2v4", 2)); }
 TEST(3) { CHECK(RunTest("l2v4", 3)); }
 TEST(4) { CHECK(RunTest("l2v4", 4)); }
 TEST(5) { CHECK(RunTest("l2v4", 5)); }
 TEST(6) { CHECK(RunTest("l2v4", 6)); }
 TEST(7) { CHECK(RunTest("l2v4", 7)); }
 TEST(8) { CHECK(RunTest("l2v4", 8)); }
 TEST(9) { CHECK(RunTest("l2v4", 9)); }
 TEST(10) { CHECK(RunTest("l2v4", 10)); }
 TEST(11) { CHECK(RunTest("l2v4", 11)); }
 TEST(12) { CHECK(RunTest("l2v4", 12)); }
 TEST(13) { CHECK(RunTest("l2v4", 13)); }
 TEST(14) { CHECK(RunTest("l2v4", 14)); }
 TEST(15) { CHECK(RunTest("l2v4", 15)); }
 TEST(16) { CHECK(RunTest("l2v4", 16)); }
 TEST(17) { CHECK(RunTest("l2v4", 17)); }
 TEST(18) { CHECK(RunTest("l2v4", 18)); }
 TEST(19) { CHECK(RunTest("l2v4", 19)); }
 TEST(20) { CHECK(RunTest("l2v4", 20)); }
 TEST(21) { CHECK(RunTest("l2v4", 21)); }
 TEST(22) { CHECK(RunTest("l2v4", 22)); }
 TEST(23) { CHECK(RunTest("l2v4", 23)); }
 TEST(24) { CHECK(RunTest("l2v4", 24)); }
 TEST(25) { CHECK(RunTest("l2v4", 25)); }
 TEST(26) { CHECK(RunTest("l2v4", 26)); }
 TEST(27) { CHECK(RunTest("l2v4", 27)); }
 TEST(28) { CHECK(RunTest("l2v4", 28)); }
 TEST(29) { CHECK(RunTest("l2v4", 29)); }
 TEST(30) { CHECK(RunTest("l2v4", 30)); }
 TEST(31) { CHECK(RunTest("l2v4", 31)); }
 TEST(32) { CHECK(RunTest("l2v4", 32)); }
 TEST(33) { CHECK(RunTest("l2v4", 33)); }

 TEST(34) { CHECK(RunTest("l2v4", 34)); }
 TEST(35) { CHECK(RunTest("l2v4", 35)); }
 TEST(36) { CHECK(RunTest("l2v4", 36)); }
 TEST(37) { CHECK(RunTest("l2v4", 37)); }
 TEST(38) { CHECK(RunTest("l2v4", 38)); }
 //TEST(39) { CHECK(RunTest("l2v4", 39)); } // AlgebraicRule
 //TEST(40) { CHECK(RunTest("l2v4", 40)); } // AlgebraicRule
 TEST(41) { CHECK(RunTest("l2v4", 41)); }
 TEST(42) { CHECK(RunTest("l2v4", 42)); }
 TEST(43) { CHECK(RunTest("l2v4", 43)); }
 TEST(44) { CHECK(RunTest("l2v4", 44)); }
 TEST(45) { CHECK(RunTest("l2v4", 45)); }
 TEST(46) { CHECK(RunTest("l2v4", 46)); }
 TEST(47) { CHECK(RunTest("l2v4", 47)); }
 TEST(48) { CHECK(RunTest("l2v4", 48)); }
 TEST(49) { CHECK(RunTest("l2v4", 49)); }
 TEST(50) { CHECK(RunTest("l2v4", 50)); }
 TEST(51) { CHECK(RunTest("l2v4", 51)); }
 TEST(52) { CHECK(RunTest("l2v4", 52)); }
 TEST(53) { CHECK(RunTest("l2v4", 53)); }
 TEST(54) { CHECK(RunTest("l2v4", 54)); }
 TEST(55) { CHECK(RunTest("l2v4", 55)); }
 TEST(56) { CHECK(RunTest("l2v4", 56)); }
 TEST(57) { CHECK(RunTest("l2v4", 57)); }
 TEST(58) { CHECK(RunTest("l2v4", 58)); }
 TEST(59) { CHECK(RunTest("l2v4", 59)); }
 TEST(60) { CHECK(RunTest("l2v4", 60)); }
 TEST(61) { CHECK(RunTest("l2v4", 61)); }
 TEST(62) { CHECK(RunTest("l2v4", 62)); }
 TEST(63) { CHECK(RunTest("l2v4", 63)); }
 TEST(64) { CHECK(RunTest("l2v4", 64)); }
 TEST(65) { CHECK(RunTest("l2v4", 65)); }
 TEST(66) { CHECK(RunTest("l2v4", 66)); }
 TEST(67) { CHECK(RunTest("l2v4", 67)); }
 TEST(68) { CHECK(RunTest("l2v4", 68)); }
 TEST(69) { CHECK(RunTest("l2v4", 69)); }
 TEST(70) { CHECK(RunTest("l2v4", 70)); }
 TEST(71) { CHECK(RunTest("l2v4", 71)); }
 TEST(72) { CHECK(RunTest("l2v4", 72)); }
 TEST(73) { CHECK(RunTest("l2v4", 73)); }
 TEST(74) { CHECK(RunTest("l2v4", 74)); }
 TEST(75) { CHECK(RunTest("l2v4", 75)); }
 TEST(76) { CHECK(RunTest("l2v4", 76)); }
 TEST(77) { CHECK(RunTest("l2v4", 77)); }
 TEST(78) { CHECK(RunTest("l2v4", 78)); }
 TEST(79) { CHECK(RunTest("l2v4", 79)); }
 TEST(80) { CHECK(RunTest("l2v4", 80)); }
 TEST(81) { CHECK(RunTest("l2v4", 81)); }
 TEST(82) { CHECK(RunTest("l2v4", 82)); }
 TEST(83) { CHECK(RunTest("l2v4", 83)); }
 TEST(84) { CHECK(RunTest("l2v4", 84)); }
 TEST(85) { CHECK(RunTest("l2v4", 85)); }
 TEST(86) { CHECK(RunTest("l2v4", 86)); }
 TEST(87) { CHECK(RunTest("l2v4", 87)); }
 TEST(88) { CHECK(RunTest("l2v4", 88)); }
 TEST(89) { CHECK(RunTest("l2v4", 89)); }
 TEST(90) { CHECK(RunTest("l2v4", 90)); }
 TEST(91) { CHECK(RunTest("l2v4", 91)); }
 TEST(92) { CHECK(RunTest("l2v4", 92)); }
 TEST(93) { CHECK(RunTest("l2v4", 93)); }
 TEST(94) { CHECK(RunTest("l2v4", 94)); }
 TEST(95) { CHECK(RunTest("l2v4", 95)); }
 TEST(96) { CHECK(RunTest("l2v4", 96)); }
 TEST(97) { CHECK(RunTest("l2v4", 97)); }
 TEST(98) { CHECK(RunTest("l2v4", 98)); }
 TEST(99) { CHECK(RunTest("l2v4", 99)); }
 TEST(100) { CHECK(RunTest("l2v4", 100)); }
 TEST(101) { CHECK(RunTest("l2v4", 101)); }
 TEST(102) { CHECK(RunTest("l2v4", 102)); }
 TEST(103) { CHECK(RunTest("l2v4", 103)); }
 TEST(104) { CHECK(RunTest("l2v4", 104)); }
 TEST(105) { CHECK(RunTest("l2v4", 105)); }
 TEST(106) { CHECK(RunTest("l2v4", 106)); }
 TEST(107) { CHECK(RunTest("l2v4", 107)); }
 TEST(108) { CHECK(RunTest("l2v4", 108)); }
 TEST(109) { CHECK(RunTest("l2v4", 109)); }
 TEST(110) { CHECK(RunTest("l2v4", 110)); }
 TEST(111) { CHECK(RunTest("l2v4", 111)); }
 TEST(112) { CHECK(RunTest("l2v4", 112)); }
 TEST(113) { CHECK(RunTest("l2v4", 113)); }
 TEST(114) { CHECK(RunTest("l2v4", 114)); }
 TEST(115) { CHECK(RunTest("l2v4", 115)); }
 TEST(116) { CHECK(RunTest("l2v4", 116)); }
 TEST(117) { CHECK(RunTest("l2v4", 117)); }
 TEST(118) { CHECK(RunTest("l2v4", 118)); }
 TEST(119) { CHECK(RunTest("l2v4", 119)); }
 TEST(120) { CHECK(RunTest("l2v4", 120)); }
 TEST(121) { CHECK(RunTest("l2v4", 121)); }
 TEST(122) { CHECK(RunTest("l2v4", 122)); }
 TEST(123) { CHECK(RunTest("l2v4", 123)); }
 TEST(124) { CHECK(RunTest("l2v4", 124)); }
 TEST(125) { CHECK(RunTest("l2v4", 125)); }
 TEST(126) { CHECK(RunTest("l2v4", 126)); }
 TEST(127) { CHECK(RunTest("l2v4", 127)); }
 TEST(128) { CHECK(RunTest("l2v4", 128)); }
 TEST(129) { CHECK(RunTest("l2v4", 129)); }
 TEST(130) { CHECK(RunTest("l2v4", 130)); }
 TEST(131) { CHECK(RunTest("l2v4", 131)); }
 TEST(132) { CHECK(RunTest("l2v4", 132)); }
 TEST(133) { CHECK(RunTest("l2v4", 133)); }
 TEST(134) { CHECK(RunTest("l2v4", 134)); }
 TEST(135) { CHECK(RunTest("l2v4", 135)); }
 TEST(136) { CHECK(RunTest("l2v4", 136)); }
 TEST(137) { CHECK(RunTest("l2v4", 137)); }
 TEST(138) { CHECK(RunTest("l2v4", 138)); }
 TEST(139) { CHECK(RunTest("l2v4", 139)); }
 TEST(140) { CHECK(RunTest("l2v4", 140)); }
 TEST(141) { CHECK(RunTest("l2v4", 141)); }
 TEST(142) { CHECK(RunTest("l2v4", 142)); }
 TEST(143) { CHECK(RunTest("l2v4", 143)); }
 TEST(144) { CHECK(RunTest("l2v4", 144)); }
 TEST(145) { CHECK(RunTest("l2v4", 145)); }
 TEST(146) { CHECK(RunTest("l2v4", 146)); }
 TEST(147) { CHECK(RunTest("l2v4", 147)); }
 TEST(148) { CHECK(RunTest("l2v4", 148)); }
 TEST(149) { CHECK(RunTest("l2v4", 149)); }
 TEST(150) { CHECK(RunTest("l2v4", 150)); }
 TEST(151) { CHECK(RunTest("l2v4", 151)); }
 TEST(152) { CHECK(RunTest("l2v4", 152)); }
 TEST(153) { CHECK(RunTest("l2v4", 153)); }
 TEST(154) { CHECK(RunTest("l2v4", 154)); }
 TEST(155) { CHECK(RunTest("l2v4", 155)); }
 TEST(156) { CHECK(RunTest("l2v4", 156)); }
 TEST(157) { CHECK(RunTest("l2v4", 157)); }
 TEST(158) { CHECK(RunTest("l2v4", 158)); }
 TEST(159) { CHECK(RunTest("l2v4", 159)); }
 TEST(160) { CHECK(RunTest("l2v4", 160)); }
 TEST(161) { CHECK(RunTest("l2v4", 161)); }
 TEST(162) { CHECK(RunTest("l2v4", 162)); }
 TEST(163) { CHECK(RunTest("l2v4", 163)); }
 TEST(164) { CHECK(RunTest("l2v4", 164)); }
 TEST(165) { CHECK(RunTest("l2v4", 165)); }
 TEST(166) { CHECK(RunTest("l2v4", 166)); }
 TEST(167) { CHECK(RunTest("l2v4", 167)); }
 TEST(168) { CHECK(RunTest("l2v4", 168)); }
 TEST(169) { CHECK(RunTest("l2v4", 169)); }
 TEST(170) { CHECK(RunTest("l2v4", 170)); }
 TEST(171) { CHECK(RunTest("l2v4", 171)); }
 TEST(172) { CHECK(RunTest("l2v4", 172)); }
 TEST(173) { CHECK(RunTest("l2v4", 173)); }
 TEST(174) { CHECK(RunTest("l2v4", 174)); }
 TEST(175) { CHECK(RunTest("l2v4", 175)); }
 TEST(176) { CHECK(RunTest("l2v4", 176)); }
 TEST(177) { CHECK(RunTest("l2v4", 177)); }
 TEST(178) { CHECK(RunTest("l2v4", 178)); }
 TEST(179) { CHECK(RunTest("l2v4", 179)); }
 TEST(180) { CHECK(RunTest("l2v4", 180)); }
 TEST(181) { CHECK(RunTest("l2v4", 181)); }
 //TEST(182) { CHECK(RunTest("l2v4", 182)); } // AlgebraicRule
 TEST(183) { CHECK(RunTest("l2v4", 183)); }
 //TEST(184) { CHECK(RunTest("l2v4", 184)); } // AlgebraicRule
 TEST(185) { CHECK(RunTest("l2v4", 185)); }
 TEST(186) { CHECK(RunTest("l2v4", 186)); }
 TEST(187) { CHECK(RunTest("l2v4", 187)); }
 TEST(188) { CHECK(RunTest("l2v4", 188)); }
 TEST(189) { CHECK(RunTest("l2v4", 189)); }
 TEST(190) { CHECK(RunTest("l2v4", 190)); }
 TEST(191) { CHECK(RunTest("l2v4", 191)); }
 TEST(192) { CHECK(RunTest("l2v4", 192)); }
 TEST(193) { CHECK(RunTest("l2v4", 193)); }
 TEST(194) { CHECK(RunTest("l2v4", 194)); }
 TEST(195) { CHECK(RunTest("l2v4", 195)); }
 TEST(196) { CHECK(RunTest("l2v4", 196)); }
 TEST(197) { CHECK(RunTest("l2v4", 197)); }
 TEST(198) { CHECK(RunTest("l2v4", 198)); }
 TEST(199) { CHECK(RunTest("l2v4", 199)); }
 TEST(200) { CHECK(RunTest("l2v4", 200)); }
 TEST(201) { CHECK(RunTest("l2v4", 201)); }
 TEST(202) { CHECK(RunTest("l2v4", 202)); }
 TEST(203) { CHECK(RunTest("l2v4", 203)); }
 TEST(204) { CHECK(RunTest("l2v4", 204)); }
 TEST(205) { CHECK(RunTest("l2v4", 205)); }
 TEST(206) { CHECK(RunTest("l2v4", 206)); }
 TEST(207) { CHECK(RunTest("l2v4", 207)); }
 TEST(208) { CHECK(RunTest("l2v4", 208)); }
 TEST(209) { CHECK(RunTest("l2v4", 209)); }
 TEST(210) { CHECK(RunTest("l2v4", 210)); }
 TEST(211) { CHECK(RunTest("l2v4", 211)); }
 TEST(212) { CHECK(RunTest("l2v4", 212)); }
 TEST(213) { CHECK(RunTest("l2v4", 213)); }
 TEST(214) { CHECK(RunTest("l2v4", 214)); }
 TEST(215) { CHECK(RunTest("l2v4", 215)); }
 TEST(216) { CHECK(RunTest("l2v4", 216)); }
 TEST(217) { CHECK(RunTest("l2v4", 217)); }
 TEST(218) { CHECK(RunTest("l2v4", 218)); }
 TEST(219) { CHECK(RunTest("l2v4", 219)); }
 TEST(220) { CHECK(RunTest("l2v4", 220)); }
 TEST(221) { CHECK(RunTest("l2v4", 221)); }
 TEST(222) { CHECK(RunTest("l2v4", 222)); }
 TEST(223) { CHECK(RunTest("l2v4", 223)); }
 TEST(224) { CHECK(RunTest("l2v4", 224)); }
 TEST(225) { CHECK(RunTest("l2v4", 225)); }
 TEST(226) { CHECK(RunTest("l2v4", 226)); }
 TEST(227) { CHECK(RunTest("l2v4", 227)); }
 TEST(228) { CHECK(RunTest("l2v4", 228)); }
 TEST(229) { CHECK(RunTest("l2v4", 229)); }
 TEST(230) { CHECK(RunTest("l2v4", 230)); }
 TEST(231) { CHECK(RunTest("l2v4", 231)); }
 TEST(232) { CHECK(RunTest("l2v4", 232)); }
 TEST(233) { CHECK(RunTest("l2v4", 233)); }
 TEST(234) { CHECK(RunTest("l2v4", 234)); }
 TEST(235) { CHECK(RunTest("l2v4", 235)); }
 TEST(236) { CHECK(RunTest("l2v4", 236)); }
 TEST(237) { CHECK(RunTest("l2v4", 237)); }
 TEST(238) { CHECK(RunTest("l2v4", 238)); }
 TEST(239) { CHECK(RunTest("l2v4", 239)); }
 TEST(240) { CHECK(RunTest("l2v4", 240)); }
 TEST(241) { CHECK(RunTest("l2v4", 241)); }
 TEST(242) { CHECK(RunTest("l2v4", 242)); }
 TEST(243) { CHECK(RunTest("l2v4", 243)); }
 TEST(244) { CHECK(RunTest("l2v4", 244)); }
 TEST(245) { CHECK(RunTest("l2v4", 245)); }
 TEST(246) { CHECK(RunTest("l2v4", 246)); }
 TEST(247) { CHECK(RunTest("l2v4", 247)); }
 TEST(248) { CHECK(RunTest("l2v4", 248)); }
 TEST(249) { CHECK(RunTest("l2v4", 249)); }
 TEST(250) { CHECK(RunTest("l2v4", 250)); }
 TEST(251) { CHECK(RunTest("l2v4", 251)); }
 TEST(252) { CHECK(RunTest("l2v4", 252)); }
 TEST(253) { CHECK(RunTest("l2v4", 253)); }
 TEST(254) { CHECK(RunTest("l2v4", 254)); }
 TEST(255) { CHECK(RunTest("l2v4", 255)); }
 TEST(256) { CHECK(RunTest("l2v4", 256)); }
 TEST(257) { CHECK(RunTest("l2v4", 257)); }
 TEST(258) { CHECK(RunTest("l2v4", 258)); }
 TEST(259) { CHECK(RunTest("l2v4", 259)); }
 TEST(260) { CHECK(RunTest("l2v4", 260)); }
 TEST(261) { CHECK(RunTest("l2v4", 261)); }
 TEST(262) { CHECK(RunTest("l2v4", 262)); }
 TEST(263) { CHECK(RunTest("l2v4", 263)); }
 TEST(264) { CHECK(RunTest("l2v4", 264)); }
 TEST(265) { CHECK(RunTest("l2v4", 265)); }
 TEST(266) { CHECK(RunTest("l2v4", 266)); }
 TEST(267) { CHECK(RunTest("l2v4", 267)); }
 TEST(268) { CHECK(RunTest("l2v4", 268)); }
 TEST(269) { CHECK(RunTest("l2v4", 269)); }
 TEST(270) { CHECK(RunTest("l2v4", 270)); }
 TEST(271) { CHECK(RunTest("l2v4", 271)); }
 TEST(272) { CHECK(RunTest("l2v4", 272)); }
 TEST(273) { CHECK(RunTest("l2v4", 273)); }
 TEST(274) { CHECK(RunTest("l2v4", 274)); }
 TEST(275) { CHECK(RunTest("l2v4", 275)); }
 TEST(276) { CHECK(RunTest("l2v4", 276)); }
 TEST(277) { CHECK(RunTest("l2v4", 277)); }
 TEST(278) { CHECK(RunTest("l2v4", 278)); }
 TEST(279) { CHECK(RunTest("l2v4", 279)); }
 TEST(280) { CHECK(RunTest("l2v4", 280)); }
 TEST(281) { CHECK(RunTest("l2v4", 281)); }
 TEST(282) { CHECK(RunTest("l2v4", 282)); }
 TEST(283) { CHECK(RunTest("l2v4", 283)); }
 TEST(284) { CHECK(RunTest("l2v4", 284)); }
 TEST(285) { CHECK(RunTest("l2v4", 285)); }
 TEST(286) { CHECK(RunTest("l2v4", 286)); }
 TEST(287) { CHECK(RunTest("l2v4", 287)); }
 TEST(288) { CHECK(RunTest("l2v4", 288)); }
 TEST(289) { CHECK(RunTest("l2v4", 289)); }
 TEST(290) { CHECK(RunTest("l2v4", 290)); }
 TEST(291) { CHECK(RunTest("l2v4", 291)); }
 TEST(292) { CHECK(RunTest("l2v4", 292)); }
 TEST(293) { CHECK(RunTest("l2v4", 293)); }
 TEST(294) { CHECK(RunTest("l2v4", 294)); }
 TEST(295) { CHECK(RunTest("l2v4", 295)); }
 TEST(296) { CHECK(RunTest("l2v4", 296)); }
 TEST(297) { CHECK(RunTest("l2v4", 297)); }
 TEST(298) { CHECK(RunTest("l2v4", 298)); }
 TEST(299) { CHECK(RunTest("l2v4", 299)); }
 TEST(300) { CHECK(RunTest("l2v4", 300)); }
 TEST(301) { CHECK(RunTest("l2v4", 301)); }
 TEST(302) { CHECK(RunTest("l2v4", 302)); }
 TEST(303) { CHECK(RunTest("l2v4", 303)); }
 TEST(304) { CHECK(RunTest("l2v4", 304)); }
 TEST(305) { CHECK(RunTest("l2v4", 305)); }
 TEST(306) { CHECK(RunTest("l2v4", 306)); }
 TEST(307) { CHECK(RunTest("l2v4", 307)); }
 TEST(308) { CHECK(RunTest("l2v4", 308)); }
 TEST(309) { CHECK(RunTest("l2v4", 309)); }
 TEST(310) { CHECK(RunTest("l2v4", 310)); }
 TEST(311) { CHECK(RunTest("l2v4", 311)); }
 TEST(312) { CHECK(RunTest("l2v4", 312)); }
 TEST(313) { CHECK(RunTest("l2v4", 313)); }
 TEST(314) { CHECK(RunTest("l2v4", 314)); }
 TEST(315) { CHECK(RunTest("l2v4", 315)); }
 TEST(316) { CHECK(RunTest("l2v4", 316)); }
 TEST(317) { CHECK(RunTest("l2v4", 317)); }
 TEST(318) { CHECK(RunTest("l2v4", 318)); }
 TEST(319) { CHECK(RunTest("l2v4", 319)); }
 TEST(320) { CHECK(RunTest("l2v4", 320)); }
 TEST(321) { CHECK(RunTest("l2v4", 321)); }
 TEST(322) { CHECK(RunTest("l2v4", 322)); }
 TEST(323) { CHECK(RunTest("l2v4", 323)); }
 TEST(324) { CHECK(RunTest("l2v4", 324)); }
 TEST(325) { CHECK(RunTest("l2v4", 325)); }
 TEST(326) { CHECK(RunTest("l2v4", 326)); }
 TEST(327) { CHECK(RunTest("l2v4", 327)); }
 TEST(328) { CHECK(RunTest("l2v4", 328)); }
 TEST(329) { CHECK(RunTest("l2v4", 329)); }
 TEST(330) { CHECK(RunTest("l2v4", 330)); }
 TEST(331) { CHECK(RunTest("l2v4", 331)); }
 TEST(332) { CHECK(RunTest("l2v4", 332)); }
 TEST(333) { CHECK(RunTest("l2v4", 333)); }
 TEST(334) { CHECK(RunTest("l2v4", 334)); }
 TEST(335) { CHECK(RunTest("l2v4", 335)); }
 TEST(336) { CHECK(RunTest("l2v4", 336)); }
 TEST(337) { CHECK(RunTest("l2v4", 337)); }
 TEST(338) { CHECK(RunTest("l2v4", 338)); }
 TEST(339) { CHECK(RunTest("l2v4", 339)); }
 TEST(340) { CHECK(RunTest("l2v4", 340)); }
 TEST(341) { CHECK(RunTest("l2v4", 341)); }
 TEST(342) { CHECK(RunTest("l2v4", 342)); }
 TEST(343) { CHECK(RunTest("l2v4", 343)); }
 TEST(344) { CHECK(RunTest("l2v4", 344)); }
 TEST(345) { CHECK(RunTest("l2v4", 345)); }
 TEST(346) { CHECK(RunTest("l2v4", 346)); }
 TEST(347) { CHECK(RunTest("l2v4", 347)); }
 TEST(348) { CHECK(RunTest("l2v4", 348)); }
 TEST(349) { CHECK(RunTest("l2v4", 349)); }
 TEST(350) { CHECK(RunTest("l2v4", 350)); }
 TEST(351) { CHECK(RunTest("l2v4", 351)); }
 TEST(352) { CHECK(RunTest("l2v4", 352)); }
 TEST(353) { CHECK(RunTest("l2v4", 353)); }
 TEST(354) { CHECK(RunTest("l2v4", 354)); }
 TEST(355) { CHECK(RunTest("l2v4", 355)); }
 TEST(356) { CHECK(RunTest("l2v4", 356)); }
 TEST(357) { CHECK(RunTest("l2v4", 357)); }
 TEST(358) { CHECK(RunTest("l2v4", 358)); }
 TEST(359) { CHECK(RunTest("l2v4", 359)); }
 TEST(360) { CHECK(RunTest("l2v4", 360)); }
 TEST(361) { CHECK(RunTest("l2v4", 361)); }
 TEST(362) { CHECK(RunTest("l2v4", 362)); }
 TEST(363) { CHECK(RunTest("l2v4", 363)); }
 TEST(364) { CHECK(RunTest("l2v4", 364)); }
 TEST(365) { CHECK(RunTest("l2v4", 365)); }
 TEST(366) { CHECK(RunTest("l2v4", 366)); }
 TEST(367) { CHECK(RunTest("l2v4", 367)); }
 TEST(368) { CHECK(RunTest("l2v4", 368)); }
 TEST(369) { CHECK(RunTest("l2v4", 369)); }
 TEST(370) { CHECK(RunTest("l2v4", 370)); }
 TEST(371) { CHECK(RunTest("l2v4", 371)); }
 TEST(372) { CHECK(RunTest("l2v4", 372)); }
 TEST(373) { CHECK(RunTest("l2v4", 373)); }
 TEST(374) { CHECK(RunTest("l2v4", 374)); }
 TEST(375) { CHECK(RunTest("l2v4", 375)); }
 TEST(376) { CHECK(RunTest("l2v4", 376)); }
 TEST(377) { CHECK(RunTest("l2v4", 377)); }
 TEST(378) { CHECK(RunTest("l2v4", 378)); }
 TEST(379) { CHECK(RunTest("l2v4", 379)); }
 TEST(380) { CHECK(RunTest("l2v4", 380)); }
 TEST(381) { CHECK(RunTest("l2v4", 381)); }
 TEST(382) { CHECK(RunTest("l2v4", 382)); }
 TEST(383) { CHECK(RunTest("l2v4", 383)); }
 TEST(384) { CHECK(RunTest("l2v4", 384)); }
 TEST(385) { CHECK(RunTest("l2v4", 385)); }
 TEST(386) { CHECK(RunTest("l2v4", 386)); }
 TEST(387) { CHECK(RunTest("l2v4", 387)); }
 TEST(388) { CHECK(RunTest("l2v4", 388)); }
 TEST(389) { CHECK(RunTest("l2v4", 389)); }
 TEST(390) { CHECK(RunTest("l2v4", 390)); }
 TEST(391) { CHECK(RunTest("l2v4", 391)); }
 TEST(392) { CHECK(RunTest("l2v4", 392)); }
 TEST(393) { CHECK(RunTest("l2v4", 393)); }
 TEST(394) { CHECK(RunTest("l2v4", 394)); }
 TEST(395) { CHECK(RunTest("l2v4", 395)); }
 TEST(396) { CHECK(RunTest("l2v4", 396)); }
 TEST(397) { CHECK(RunTest("l2v4", 397)); }
 TEST(398) { CHECK(RunTest("l2v4", 398)); }
 TEST(399) { CHECK(RunTest("l2v4", 399)); }
 TEST(400) { CHECK(RunTest("l2v4", 400)); }
 TEST(401) { CHECK(RunTest("l2v4", 401)); }
 TEST(402) { CHECK(RunTest("l2v4", 402)); }
 TEST(403) { CHECK(RunTest("l2v4", 403)); }
 TEST(404) { CHECK(RunTest("l2v4", 404)); }
 TEST(405) { CHECK(RunTest("l2v4", 405)); }
 TEST(406) { CHECK(RunTest("l2v4", 406)); }
 TEST(407) { CHECK(RunTest("l2v4", 407)); }
 TEST(408) { CHECK(RunTest("l2v4", 408)); }
 TEST(409) { CHECK(RunTest("l2v4", 409)); }
 TEST(410) { CHECK(RunTest("l2v4", 410)); }
 TEST(411) { CHECK(RunTest("l2v4", 411)); }
 TEST(412) { CHECK(RunTest("l2v4", 412)); }
 TEST(413) { CHECK(RunTest("l2v4", 413)); }
 TEST(414) { CHECK(RunTest("l2v4", 414)); }
 TEST(415) { CHECK(RunTest("l2v4", 415)); }
 TEST(416) { CHECK(RunTest("l2v4", 416)); }
 TEST(417) { CHECK(RunTest("l2v4", 417)); }
 TEST(418) { CHECK(RunTest("l2v4", 418)); }
 TEST(419) { CHECK(RunTest("l2v4", 419)); }
 TEST(420) { CHECK(RunTest("l2v4", 420)); }
 TEST(421) { CHECK(RunTest("l2v4", 421)); }
 TEST(422) { CHECK(RunTest("l2v4", 422)); }
 TEST(423) { CHECK(RunTest("l2v4", 423)); }
 TEST(424) { CHECK(RunTest("l2v4", 424)); }
 TEST(425) { CHECK(RunTest("l2v4", 425)); }
 TEST(426) { CHECK(RunTest("l2v4", 426)); }
 TEST(427) { CHECK(RunTest("l2v4", 427)); }
 TEST(428) { CHECK(RunTest("l2v4", 428)); }
 TEST(429) { CHECK(RunTest("l2v4", 429)); }
 TEST(430) { CHECK(RunTest("l2v4", 430)); }
 TEST(431) { CHECK(RunTest("l2v4", 431)); }
 TEST(432) { CHECK(RunTest("l2v4", 432)); }
 TEST(433) { CHECK(RunTest("l2v4", 433)); }
 TEST(434) { CHECK(RunTest("l2v4", 434)); }
 TEST(435) { CHECK(RunTest("l2v4", 435)); }
 TEST(436) { CHECK(RunTest("l2v4", 436)); }
 TEST(437) { CHECK(RunTest("l2v4", 437)); }
 TEST(438) { CHECK(RunTest("l2v4", 438)); }
 TEST(439) { CHECK(RunTest("l2v4", 439)); }
 TEST(440) { CHECK(RunTest("l2v4", 440)); }
 TEST(441) { CHECK(RunTest("l2v4", 441)); }
 TEST(442) { CHECK(RunTest("l2v4", 442)); }
 TEST(443) { CHECK(RunTest("l2v4", 443)); }
 TEST(444) { CHECK(RunTest("l2v4", 444)); }
 TEST(445) { CHECK(RunTest("l2v4", 445)); }
 TEST(446) { CHECK(RunTest("l2v4", 446)); }
 TEST(447) { CHECK(RunTest("l2v4", 447)); }
 TEST(448) { CHECK(RunTest("l2v4", 448)); }
 TEST(449) { CHECK(RunTest("l2v4", 449)); }
 TEST(450) { CHECK(RunTest("l2v4", 450)); }
 TEST(451) { CHECK(RunTest("l2v4", 451)); }
 TEST(452) { CHECK(RunTest("l2v4", 452)); }
 TEST(453) { CHECK(RunTest("l2v4", 453)); }
 TEST(454) { CHECK(RunTest("l2v4", 454)); }
 TEST(455) { CHECK(RunTest("l2v4", 455)); }
 TEST(456) { CHECK(RunTest("l2v4", 456)); }
 TEST(457) { CHECK(RunTest("l2v4", 457)); }
 TEST(458) { CHECK(RunTest("l2v4", 458)); }
 TEST(459) { CHECK(RunTest("l2v4", 459)); }
 TEST(460) { CHECK(RunTest("l2v4", 460)); }
 TEST(461) { CHECK(RunTest("l2v4", 461)); }
 TEST(462) { CHECK(RunTest("l2v4", 462)); }
 TEST(463) { CHECK(RunTest("l2v4", 463)); }
 TEST(464) { CHECK(RunTest("l2v4", 464)); }
 TEST(465) { CHECK(RunTest("l2v4", 465)); }
 TEST(466) { CHECK(RunTest("l2v4", 466)); }
 TEST(467) { CHECK(RunTest("l2v4", 467)); }
 TEST(468) { CHECK(RunTest("l2v4", 468)); }
 TEST(469) { CHECK(RunTest("l2v4", 469)); }
 TEST(470) { CHECK(RunTest("l2v4", 470)); }
 TEST(471) { CHECK(RunTest("l2v4", 471)); }
 TEST(472) { CHECK(RunTest("l2v4", 472)); }
 TEST(473) { CHECK(RunTest("l2v4", 473)); }
 TEST(474) { CHECK(RunTest("l2v4", 474)); }
 TEST(475) { CHECK(RunTest("l2v4", 475)); }
 TEST(476) { CHECK(RunTest("l2v4", 476)); }
 TEST(477) { CHECK(RunTest("l2v4", 477)); }
 TEST(478) { CHECK(RunTest("l2v4", 478)); }
 TEST(479) { CHECK(RunTest("l2v4", 479)); }
 TEST(480) { CHECK(RunTest("l2v4", 480)); }
 TEST(481) { CHECK(RunTest("l2v4", 481)); }
 TEST(482) { CHECK(RunTest("l2v4", 482)); }
 TEST(483) { CHECK(RunTest("l2v4", 483)); }
 TEST(484) { CHECK(RunTest("l2v4", 484)); }
 TEST(485) { CHECK(RunTest("l2v4", 485)); }
 TEST(486) { CHECK(RunTest("l2v4", 486)); }
 TEST(487) { CHECK(RunTest("l2v4", 487)); }
 TEST(488) { CHECK(RunTest("l2v4", 488)); }
 TEST(489) { CHECK(RunTest("l2v4", 489)); }
 TEST(490) { CHECK(RunTest("l2v4", 490)); }
 TEST(491) { CHECK(RunTest("l2v4", 491)); }
 TEST(492) { CHECK(RunTest("l2v4", 492)); }
 TEST(493) { CHECK(RunTest("l2v4", 493)); }
 TEST(494) { CHECK(RunTest("l2v4", 494)); }
 TEST(495) { CHECK(RunTest("l2v4", 495)); }
 TEST(496) { CHECK(RunTest("l2v4", 496)); }
 TEST(497) { CHECK(RunTest("l2v4", 497)); }
 TEST(498) { CHECK(RunTest("l2v4", 498)); }
 TEST(499) { CHECK(RunTest("l2v4", 499)); }
 TEST(500) { CHECK(RunTest("l2v4", 500)); }
 TEST(501) { CHECK(RunTest("l2v4", 501)); }
 TEST(502) { CHECK(RunTest("l2v4", 502)); }
 TEST(503) { CHECK(RunTest("l2v4", 503)); }
 TEST(504) { CHECK(RunTest("l2v4", 504)); }
 TEST(505) { CHECK(RunTest("l2v4", 505)); }
 TEST(506) { CHECK(RunTest("l2v4", 506)); }
 TEST(507) { CHECK(RunTest("l2v4", 507)); }
 TEST(508) { CHECK(RunTest("l2v4", 508)); }
 TEST(509) { CHECK(RunTest("l2v4", 509)); }
 TEST(510) { CHECK(RunTest("l2v4", 510)); }
 TEST(511) { CHECK(RunTest("l2v4", 511)); }
 TEST(512) { CHECK(RunTest("l2v4", 512)); }
 TEST(513) { CHECK(RunTest("l2v4", 513)); }
 TEST(514) { CHECK(RunTest("l2v4", 514)); }
 TEST(515) { CHECK(RunTest("l2v4", 515)); }
 TEST(516) { CHECK(RunTest("l2v4", 516)); }
 TEST(517) { CHECK(RunTest("l2v4", 517)); }
 TEST(518) { CHECK(RunTest("l2v4", 518)); }
 TEST(519) { CHECK(RunTest("l2v4", 519)); }
 TEST(520) { CHECK(RunTest("l2v4", 520)); }
 TEST(521) { CHECK(RunTest("l2v4", 521)); }
 TEST(522) { CHECK(RunTest("l2v4", 522)); }
 TEST(523) { CHECK(RunTest("l2v4", 523)); }
 TEST(524) { CHECK(RunTest("l2v4", 524)); }
 TEST(525) { CHECK(RunTest("l2v4", 525)); }
 TEST(526) { CHECK(RunTest("l2v4", 526)); }
 TEST(527) { CHECK(RunTest("l2v4", 527)); }
 TEST(528) { CHECK(RunTest("l2v4", 528)); }
 TEST(529) { CHECK(RunTest("l2v4", 529)); }
 TEST(530) { CHECK(RunTest("l2v4", 530)); }
 //TEST(531) { CHECK(RunTest("l2v4", 531)); } // AlgebraicRule
 TEST(532) { CHECK(RunTest("l2v4", 532)); }
 //TEST(533) { CHECK(RunTest("l2v4", 533)); } // AlgebraicRule
 //TEST(534) { CHECK(RunTest("l2v4", 534)); } // AlgebraicRule
 //TEST(535) { CHECK(RunTest("l2v4", 535)); } // AlgebraicRule
 //TEST(536) { CHECK(RunTest("l2v4", 536)); } // AlgebraicRule
 //TEST(537) { CHECK(RunTest("l2v4", 537)); } // AlgebraicRule
 //TEST(538) { CHECK(RunTest("l2v4", 538)); } // AlgebraicRule
 TEST(539) { CHECK(RunTest("l2v4", 539)); }
 TEST(540) { CHECK(RunTest("l2v4", 540)); }
 TEST(541) { CHECK(RunTest("l2v4", 541)); }
 TEST(542) { CHECK(RunTest("l2v4", 542)); }
 //TEST(543) { CHECK(RunTest("l2v4", 543)); } // AlgebraicRule
 TEST(544) { CHECK(RunTest("l2v4", 544)); }
 TEST(545) { CHECK(RunTest("l2v4", 545)); }
 //TEST(546) { CHECK(RunTest("l2v4", 546)); } // AlgebraicRule
 TEST(547) { CHECK(RunTest("l2v4", 547)); }
 //TEST(548) { CHECK(RunTest("l2v4", 548)); } // AlgebraicRule
 //TEST(549) { CHECK(RunTest("l2v4", 549)); } // AlgebraicRule
 //TEST(550) { CHECK(RunTest("l2v4", 550)); } // AlgebraicRule
 //TEST(551) { CHECK(RunTest("l2v4", 551)); } // AlgebraicRule
 //TEST(552) { CHECK(RunTest("l2v4", 552)); } // AlgebraicRule
 //TEST(553) { CHECK(RunTest("l2v4", 553)); } // AlgebraicRule
 //TEST(554) { CHECK(RunTest("l2v4", 554)); } // AlgebraicRule
 //TEST(555) { CHECK(RunTest("l2v4", 555)); } // AlgebraicRule
 //TEST(556) { CHECK(RunTest("l2v4", 556)); } // AlgebraicRule
 //TEST(557) { CHECK(RunTest("l2v4", 557)); } // AlgebraicRule
 //TEST(558) { CHECK(RunTest("l2v4", 558)); } // AlgebraicRule
 //TEST(559) { CHECK(RunTest("l2v4", 559)); } // AlgebraicRule
 //TEST(560) { CHECK(RunTest("l2v4", 560)); } // AlgebraicRule
 //TEST(561) { CHECK(RunTest("l2v4", 561)); } // AlgebraicRule
 //TEST(562) { CHECK(RunTest("l2v4", 562)); } // AlgebraicRule
 //TEST(563) { CHECK(RunTest("l2v4", 563)); } // AlgebraicRule
 //TEST(564) { CHECK(RunTest("l2v4", 564)); } // AlgebraicRule
 //TEST(565) { CHECK(RunTest("l2v4", 565)); } // AlgebraicRule
 //TEST(566) { CHECK(RunTest("l2v4", 566)); } // AlgebraicRule
 //TEST(567) { CHECK(RunTest("l2v4", 567)); } // AlgebraicRule
 TEST(568) { CHECK(RunTest("l2v4", 568)); }
 //TEST(569) { CHECK(RunTest("l2v4", 569)); } // AlgebraicRule
 //TEST(570) { CHECK(RunTest("l2v4", 570)); } // AlgebraicRule
 //TEST(571) { CHECK(RunTest("l2v4", 571)); } // AlgebraicRule
 TEST(572) { CHECK(RunTest("l2v4", 572)); }
 //TEST(573) { CHECK(RunTest("l2v4", 573)); } // AlgebraicRule
 TEST(574) { CHECK(RunTest("l2v4", 574)); }
 //TEST(575) { CHECK(RunTest("l2v4", 575)); } // AlgebraicRule
 //TEST(576) { CHECK(RunTest("l2v4", 576)); } // AlgebraicRule
 TEST(577) { CHECK(RunTest("l2v4", 577)); }
 TEST(578) { CHECK(RunTest("l2v4", 578)); }
 TEST(579) { CHECK(RunTest("l2v4", 579)); }
 TEST(580) { CHECK(RunTest("l2v4", 580)); }
 TEST(581) { CHECK(RunTest("l2v4", 581)); }
 TEST(582) { CHECK(RunTest("l2v4", 582)); }
 TEST(583) { CHECK(RunTest("l2v4", 583)); }
 TEST(584) { CHECK(RunTest("l2v4", 584)); }
 TEST(585) { CHECK(RunTest("l2v4", 585)); }
 TEST(586) { CHECK(RunTest("l2v4", 586)); }
 TEST(587) { CHECK(RunTest("l2v4", 587)); }
 TEST(588) { CHECK(RunTest("l2v4", 588)); }
 TEST(589) { CHECK(RunTest("l2v4", 589)); }
 TEST(590) { CHECK(RunTest("l2v4", 590)); }
 TEST(591) { CHECK(RunTest("l2v4", 591)); }
 TEST(592) { CHECK(RunTest("l2v4", 592)); }
 TEST(593) { CHECK(RunTest("l2v4", 593)); }
 TEST(594) { CHECK(RunTest("l2v4", 594)); }
 TEST(595) { CHECK(RunTest("l2v4", 595)); }
 TEST(596) { CHECK(RunTest("l2v4", 596)); }
 TEST(597) { CHECK(RunTest("l2v4", 597)); }
 TEST(598) { CHECK(RunTest("l2v4", 598)); }
 TEST(599) { CHECK(RunTest("l2v4", 599)); }
 TEST(600) { CHECK(RunTest("l2v4", 600)); }
 TEST(601) { CHECK(RunTest("l2v4", 601)); }
 TEST(602) { CHECK(RunTest("l2v4", 602)); }
 TEST(603) { CHECK(RunTest("l2v4", 603)); }
 TEST(604) { CHECK(RunTest("l2v4", 604)); }
 TEST(605) { CHECK(RunTest("l2v4", 605)); }
 TEST(606) { CHECK(RunTest("l2v4", 606)); }
 TEST(607) { CHECK(RunTest("l2v4", 607)); }
 TEST(608) { CHECK(RunTest("l2v4", 608)); }
 TEST(609) { CHECK(RunTest("l2v4", 609)); }
 TEST(610) { CHECK(RunTest("l2v4", 610)); }
 TEST(611) { CHECK(RunTest("l2v4", 611)); }
 TEST(612) { CHECK(RunTest("l2v4", 612)); }
 //TEST(613) { CHECK(RunTest("l2v4", 613)); } // AlgebraicRule
 //TEST(614) { CHECK(RunTest("l2v4", 614)); } // AlgebraicRule
 //TEST(615) { CHECK(RunTest("l2v4", 615)); } // AlgebraicRule
 TEST(616) { CHECK(RunTest("l2v4", 616)); }
 TEST(617) { CHECK(RunTest("l2v4", 617)); }
 TEST(618) { CHECK(RunTest("l2v4", 618)); }
 TEST(619) { CHECK(RunTest("l2v4", 619)); }
 TEST(620) { CHECK(RunTest("l2v4", 620)); }
 TEST(621) { CHECK(RunTest("l2v4", 621)); }
 TEST(622) { CHECK(RunTest("l2v4", 622)); }
 TEST(623) { CHECK(RunTest("l2v4", 623)); }
 TEST(624) { CHECK(RunTest("l2v4", 624)); }
 TEST(625) { CHECK(RunTest("l2v4", 625)); }
 TEST(626) { CHECK(RunTest("l2v4", 626)); }
 TEST(627) { CHECK(RunTest("l2v4", 627)); }
 //TEST(628) { CHECK(RunTest("l2v4", 628)); } // AlgebraicRule
 //TEST(629) { CHECK(RunTest("l2v4", 629)); } // AlgebraicRule
 //TEST(630) { CHECK(RunTest("l2v4", 630)); } // AlgebraicRule
 TEST(631) { CHECK(RunTest("l2v4", 631)); }
 TEST(632) { CHECK(RunTest("l2v4", 632)); }
 TEST(633) { CHECK(RunTest("l2v4", 633)); }
 TEST(634) { CHECK(RunTest("l2v4", 634)); }
 TEST(635) { CHECK(RunTest("l2v4", 635)); }
 TEST(636) { CHECK(RunTest("l2v4", 636)); }
 TEST(637) { CHECK(RunTest("l2v4", 637)); }
 TEST(638) { CHECK(RunTest("l2v4", 638)); }
 TEST(639) { CHECK(RunTest("l2v4", 639)); }
 TEST(640) { CHECK(RunTest("l2v4", 640)); }
 TEST(641) { CHECK(RunTest("l2v4", 641)); }
 TEST(642) { CHECK(RunTest("l2v4", 642)); }
 TEST(643) { CHECK(RunTest("l2v4", 643)); }
 TEST(644) { CHECK(RunTest("l2v4", 644)); }
 TEST(645) { CHECK(RunTest("l2v4", 645)); }
 TEST(646) { CHECK(RunTest("l2v4", 646)); }
 TEST(647) { CHECK(RunTest("l2v4", 647)); }
 TEST(648) { CHECK(RunTest("l2v4", 648)); }
 TEST(649) { CHECK(RunTest("l2v4", 649)); }
 TEST(650) { CHECK(RunTest("l2v4", 650)); }
 TEST(651) { CHECK(RunTest("l2v4", 651)); }
 TEST(652) { CHECK(RunTest("l2v4", 652)); }
 TEST(653) { CHECK(RunTest("l2v4", 653)); }
 TEST(654) { CHECK(RunTest("l2v4", 654)); }
 TEST(655) { CHECK(RunTest("l2v4", 655)); }
 TEST(656) { CHECK(RunTest("l2v4", 656)); }
 TEST(657) { CHECK(RunTest("l2v4", 657)); }
 //TEST(658) { CHECK(RunTest("l2v4", 658)); } // AlgebraicRule
 //TEST(659) { CHECK(RunTest("l2v4", 659)); } // AlgebraicRule
 //TEST(660) { CHECK(RunTest("l2v4", 660)); } // AlgebraicRule
 //TEST(661) { CHECK(RunTest("l2v4", 661)); } // AlgebraicRule
 //TEST(662) { CHECK(RunTest("l2v4", 662)); } // AlgebraicRule
 //TEST(663) { CHECK(RunTest("l2v4", 663)); } // AlgebraicRule
 //TEST(664) { CHECK(RunTest("l2v4", 664)); } // AlgebraicRule
 //TEST(665) { CHECK(RunTest("l2v4", 665)); } // AlgebraicRule
 //TEST(666) { CHECK(RunTest("l2v4", 666)); } // AlgebraicRule
 TEST(667) { CHECK(RunTest("l2v4", 667)); }
 TEST(668) { CHECK(RunTest("l2v4", 668)); }
 TEST(669) { CHECK(RunTest("l2v4", 669)); }
 TEST(670) { CHECK(RunTest("l2v4", 670)); }
 TEST(671) { CHECK(RunTest("l2v4", 671)); }
 TEST(672) { CHECK(RunTest("l2v4", 672)); }
 //TEST(673) { CHECK(RunTest("l2v4", 673)); } // AlgebraicRule
 //TEST(674) { CHECK(RunTest("l2v4", 674)); } // AlgebraicRule
 //TEST(675) { CHECK(RunTest("l2v4", 675)); } // AlgebraicRule
 TEST(676) { CHECK(RunTest("l2v4", 676)); }
 TEST(677) { CHECK(RunTest("l2v4", 677)); }
 TEST(678) { CHECK(RunTest("l2v4", 678)); }
 TEST(679) { CHECK(RunTest("l2v4", 679)); }
 TEST(680) { CHECK(RunTest("l2v4", 680)); }
 TEST(681) { CHECK(RunTest("l2v4", 681)); }
 TEST(682) { CHECK(RunTest("l2v4", 682)); }
 TEST(683) { CHECK(RunTest("l2v4", 683)); }
 TEST(684) { CHECK(RunTest("l2v4", 684)); }
 TEST(685) { CHECK(RunTest("l2v4", 685)); }
 TEST(686) { CHECK(RunTest("l2v4", 686)); }
 //TEST(687) { CHECK(RunTest("l2v4", 687)); } // AlgebraicRule
 TEST(688) { CHECK(RunTest("l2v4", 688)); }
 TEST(689) { CHECK(RunTest("l2v4", 689)); }
 TEST(690) { CHECK(RunTest("l2v4", 690)); }
 TEST(691) { CHECK(RunTest("l2v4", 691)); }
 TEST(692) { CHECK(RunTest("l2v4", 692)); }
 TEST(693) { CHECK(RunTest("l2v4", 693)); }
 TEST(694) { CHECK(RunTest("l2v4", 694)); }
 //TEST(695) { CHECK(RunTest("l2v4", 695)); } // AlgebraicRule
 //TEST(696) { CHECK(RunTest("l2v4", 696)); } // AlgebraicRule
 TEST(697) { CHECK(RunTest("l2v4", 697)); }
 TEST(698) { CHECK(RunTest("l2v4", 698)); }
 TEST(699) { CHECK(RunTest("l2v4", 699)); }
 TEST(700) { CHECK(RunTest("l2v4", 700)); }
 TEST(701) { CHECK(RunTest("l2v4", 701)); }
 TEST(702) { CHECK(RunTest("l2v4", 702)); }
 TEST(703) { CHECK(RunTest("l2v4", 703)); }
 TEST(704) { CHECK(RunTest("l2v4", 704)); }
 //TEST(705) { CHECK(RunTest("l2v4", 705)); } // AlgebraicRule
 TEST(706) { CHECK(RunTest("l2v4", 706)); }
 TEST(707) { CHECK(RunTest("l2v4", 707)); }
 TEST(708) { CHECK(RunTest("l2v4", 708)); }
 TEST(709) { CHECK(RunTest("l2v4", 709)); }
 TEST(710) { CHECK(RunTest("l2v4", 710)); }
 TEST(711) { CHECK(RunTest("l2v4", 711)); }
 TEST(712) { CHECK(RunTest("l2v4", 712)); }
 TEST(713) { CHECK(RunTest("l2v4", 713)); }
 TEST(714) { CHECK(RunTest("l2v4", 714)); }
 TEST(715) { CHECK(RunTest("l2v4", 715)); }
 TEST(716) { CHECK(RunTest("l2v4", 716)); }
 TEST(717) { CHECK(RunTest("l2v4", 717)); }
 TEST(718) { CHECK(RunTest("l2v4", 718)); }
 TEST(719) { CHECK(RunTest("l2v4", 719)); }
 TEST(720) { CHECK(RunTest("l2v4", 720)); }
 TEST(721) { CHECK(RunTest("l2v4", 721)); }
 TEST(722) { CHECK(RunTest("l2v4", 722)); }
 TEST(723) { CHECK(RunTest("l2v4", 723)); }
 TEST(724) { CHECK(RunTest("l2v4", 724)); }
 TEST(725) { CHECK(RunTest("l2v4", 725)); }
 TEST(726) { CHECK(RunTest("l2v4", 726)); }
 TEST(727) { CHECK(RunTest("l2v4", 727)); }
 TEST(728) { CHECK(RunTest("l2v4", 728)); }
 TEST(729) { CHECK(RunTest("l2v4", 729)); }
 TEST(730) { CHECK(RunTest("l2v4", 730)); }
 TEST(731) { CHECK(RunTest("l2v4", 731)); }
 TEST(732) { CHECK(RunTest("l2v4", 732)); }
 TEST(733) { CHECK(RunTest("l2v4", 733)); }
 TEST(734) { CHECK(RunTest("l2v4", 734)); }
 TEST(735) { CHECK(RunTest("l2v4", 735)); }
 TEST(736) { CHECK(RunTest("l2v4", 736)); }
 TEST(737) { CHECK(RunTest("l2v4", 737)); }
 TEST(738) { CHECK(RunTest("l2v4", 738)); }
 TEST(739) { CHECK(RunTest("l2v4", 739)); }
 TEST(740) { CHECK(RunTest("l2v4", 740)); }
 TEST(741) { CHECK(RunTest("l2v4", 741)); }
 TEST(742) { CHECK(RunTest("l2v4", 742)); }
 TEST(743) { CHECK(RunTest("l2v4", 743)); }
 TEST(744) { CHECK(RunTest("l2v4", 744)); }
 TEST(745) { CHECK(RunTest("l2v4", 745)); }
 TEST(746) { CHECK(RunTest("l2v4", 746)); }
 TEST(747) { CHECK(RunTest("l2v4", 747)); }
 TEST(748) { CHECK(RunTest("l2v4", 748)); }
 TEST(749) { CHECK(RunTest("l2v4", 749)); }
 TEST(750) { CHECK(RunTest("l2v4", 750)); }
 TEST(751) { CHECK(RunTest("l2v4", 751)); }
 TEST(752) { CHECK(RunTest("l2v4", 752)); }
 TEST(753) { CHECK(RunTest("l2v4", 753)); }
 TEST(754) { CHECK(RunTest("l2v4", 754)); }
 TEST(755) { CHECK(RunTest("l2v4", 755)); }
 TEST(756) { CHECK(RunTest("l2v4", 756)); }
 TEST(757) { CHECK(RunTest("l2v4", 757)); }
 TEST(758) { CHECK(RunTest("l2v4", 758)); }
 TEST(759) { CHECK(RunTest("l2v4", 759)); }
 //TEST(760) { CHECK(RunTest("l2v4", 760)); } // AlgebraicRule
 //TEST(761) { CHECK(RunTest("l2v4", 761)); } // AlgebraicRule
 //TEST(762) { CHECK(RunTest("l2v4", 762)); } // AlgebraicRule
 TEST(763) { CHECK(RunTest("l2v4", 763)); }
 TEST(764) { CHECK(RunTest("l2v4", 764)); }
 TEST(765) { CHECK(RunTest("l2v4", 765)); }
 TEST(766) { CHECK(RunTest("l2v4", 766)); }
 TEST(767) { CHECK(RunTest("l2v4", 767)); }
 TEST(768) { CHECK(RunTest("l2v4", 768)); }
 TEST(769) { CHECK(RunTest("l2v4", 769)); }
 TEST(770) { CHECK(RunTest("l2v4", 770)); }
 TEST(771) { CHECK(RunTest("l2v4", 771)); }
 TEST(772) { CHECK(RunTest("l2v4", 772)); }
 TEST(773) { CHECK(RunTest("l2v4", 773)); }
 TEST(774) { CHECK(RunTest("l2v4", 774)); }
 TEST(775) { CHECK(RunTest("l2v4", 775)); }
 TEST(776) { CHECK(RunTest("l2v4", 776)); }
 //TEST(777) { CHECK(RunTest("l2v4", 777)); } // AlgebraicRule
 //TEST(778) { CHECK(RunTest("l2v4", 778)); } // AlgebraicRule
 //TEST(779) { CHECK(RunTest("l2v4", 779)); } // AlgebraicRule
 //TEST(780) { CHECK(RunTest("l2v4", 780)); } // AlgebraicRule
 TEST(781) { CHECK(RunTest("l2v4", 781)); }
 TEST(782) { CHECK(RunTest("l2v4", 782)); }
 TEST(783) { CHECK(RunTest("l2v4", 783)); }
 TEST(784) { CHECK(RunTest("l2v4", 784)); }
 TEST(785) { CHECK(RunTest("l2v4", 785)); }
 TEST(786) { CHECK(RunTest("l2v4", 786)); }
 TEST(787) { CHECK(RunTest("l2v4", 787)); }
 TEST(788) { CHECK(RunTest("l2v4", 788)); }
 TEST(789) { CHECK(RunTest("l2v4", 789)); }
 TEST(790) { CHECK(RunTest("l2v4", 790)); }
 TEST(791) { CHECK(RunTest("l2v4", 791)); }
 TEST(792) { CHECK(RunTest("l2v4", 792)); }
 TEST(793) { CHECK(RunTest("l2v4", 793)); }
 TEST(794) { CHECK(RunTest("l2v4", 794)); }
 TEST(795) { CHECK(RunTest("l2v4", 795)); }
 TEST(796) { CHECK(RunTest("l2v4", 796)); }
 TEST(797) { CHECK(RunTest("l2v4", 797)); }
 TEST(798) { CHECK(RunTest("l2v4", 798)); }
 TEST(799) { CHECK(RunTest("l2v4", 799)); }
 TEST(800) { CHECK(RunTest("l2v4", 800)); }
 TEST(801) { CHECK(RunTest("l2v4", 801)); }
 TEST(802) { CHECK(RunTest("l2v4", 802)); }
 TEST(803) { CHECK(RunTest("l2v4", 803)); }
 TEST(804) { CHECK(RunTest("l2v4", 804)); }
 TEST(805) { CHECK(RunTest("l2v4", 805)); }
 TEST(806) { CHECK(RunTest("l2v4", 806)); }
 TEST(807) { CHECK(RunTest("l2v4", 807)); }
 TEST(808) { CHECK(RunTest("l2v4", 808)); }
 TEST(809) { CHECK(RunTest("l2v4", 809)); }
 TEST(810) { CHECK(RunTest("l2v4", 810)); }
 TEST(811) { CHECK(RunTest("l2v4", 811)); }
 TEST(812) { CHECK(RunTest("l2v4", 812)); }
 TEST(813) { CHECK(RunTest("l2v4", 813)); }
 TEST(814) { CHECK(RunTest("l2v4", 814)); }
 TEST(815) { CHECK(RunTest("l2v4", 815)); }
 TEST(816) { CHECK(RunTest("l2v4", 816)); }
 TEST(817) { CHECK(RunTest("l2v4", 817)); }
 TEST(818) { CHECK(RunTest("l2v4", 818)); }
 TEST(819) { CHECK(RunTest("l2v4", 819)); }
 TEST(820) { CHECK(RunTest("l2v4", 820)); }
 TEST(821) { CHECK(RunTest("l2v4", 821)); }
 TEST(822) { CHECK(RunTest("l2v4", 822)); }
 TEST(823) { CHECK(RunTest("l2v4", 823)); }
 TEST(824) { CHECK(RunTest("l2v4", 824)); }
 TEST(825) { CHECK(RunTest("l2v4", 825)); }
 TEST(826) { CHECK(RunTest("l2v4", 826)); }
 TEST(827) { CHECK(RunTest("l2v4", 827)); }
 TEST(828) { CHECK(RunTest("l2v4", 828)); }
 TEST(829) { CHECK(RunTest("l2v4", 829)); }
 TEST(830) { CHECK(RunTest("l2v4", 830)); }
 TEST(831) { CHECK(RunTest("l2v4", 831)); }
 TEST(832) { CHECK(RunTest("l2v4", 832)); }
 TEST(833) { CHECK(RunTest("l2v4", 833)); }
 TEST(834) { CHECK(RunTest("l2v4", 834)); }
 TEST(835) { CHECK(RunTest("l2v4", 835)); }
 TEST(836) { CHECK(RunTest("l2v4", 836)); }
 TEST(837) { CHECK(RunTest("l2v4", 837)); }
 TEST(838) { CHECK(RunTest("l2v4", 838)); }
 TEST(839) { CHECK(RunTest("l2v4", 839)); }
 TEST(840) { CHECK(RunTest("l2v4", 840)); }
 TEST(841) { CHECK(RunTest("l2v4", 841)); }
 TEST(842) { CHECK(RunTest("l2v4", 842)); }
 TEST(843) { CHECK(RunTest("l2v4", 843)); }
//TEST(844) { CHECK(RunTest("l2v4", 844)); } // AlgebraicRule
TEST(845) { CHECK(RunTest("l2v4", 845)); }
TEST(846) { CHECK(RunTest("l2v4", 846)); }
TEST(847) { CHECK(RunTest("l2v4", 847)); }
TEST(848) { CHECK(RunTest("l2v4", 848)); }
TEST(849) { CHECK(RunTest("l2v4", 849)); }
TEST(850) { CHECK(RunTest("l2v4", 850)); }
TEST(851) { CHECK(RunTest("l2v4", 851)); }
TEST(852) { CHECK(RunTest("l2v4", 852)); }
TEST(853) { CHECK(RunTest("l2v4", 853)); }
TEST(854) { CHECK(RunTest("l2v4", 854)); }
TEST(855) { CHECK(RunTest("l2v4", 855)); }
TEST(856) { CHECK(RunTest("l2v4", 856)); }
TEST(857) { CHECK(RunTest("l2v4", 857)); }
TEST(858) { CHECK(RunTest("l2v4", 858)); }
TEST(859) { CHECK(RunTest("l2v4", 859)); }
TEST(860) { CHECK(RunTest("l2v4", 860)); }
TEST(861) { CHECK(RunTest("l2v4", 861)); }
TEST(862) { CHECK(RunTest("l2v4", 862)); }
TEST(863) { CHECK(RunTest("l2v4", 863)); }
TEST(864) { CHECK(RunTest("l2v4", 864)); }
TEST(865) { CHECK(RunTest("l2v4", 865)); }
TEST(866) { CHECK(RunTest("l2v4", 866)); }
TEST(867) { CHECK(RunTest("l2v4", 867)); }
TEST(868) { CHECK(RunTest("l2v4", 868)); }
TEST(869) { CHECK(RunTest("l2v4", 869)); }
//TEST(870) { CHECK(RunTest("l2v4", 870)); } // FastReaction
//TEST(871) { CHECK(RunTest("l2v4", 871)); } // FastReaction
//TEST(872) { CHECK(RunTest("l2v4", 872)); } // FastReaction
//TEST(873) { CHECK(RunTest("l2v4", 873)); } // FastReaction
//TEST(874) { CHECK(RunTest("l2v4", 874)); } // FastReaction
//TEST(875) { CHECK(RunTest("l2v4", 875)); } // FastReaction
//TEST(876) { CHECK(RunTest("l2v4", 876)); } // AlgebraicRule
TEST(877) { CHECK(RunTest("l2v4", 877)); }
TEST(878) { CHECK(RunTest("l2v4", 878)); }
TEST(879) { CHECK(RunTest("l2v4", 879)); }
TEST(880) { CHECK(RunTest("l2v4", 880)); }
TEST(881) { CHECK(RunTest("l2v4", 881)); }
TEST(882) { CHECK(RunTest("l2v4", 882)); }
TEST(883) { CHECK(RunTest("l2v4", 883)); }
TEST(884) { CHECK(RunTest("l2v4", 884)); }
TEST(885) { CHECK(RunTest("l2v4", 885)); }
TEST(886) { CHECK(RunTest("l2v4", 886)); }
TEST(887) { CHECK(RunTest("l2v4", 887)); }
TEST(888) { CHECK(RunTest("l2v4", 888)); }
TEST(889) { CHECK(RunTest("l2v4", 889)); }
TEST(890) { CHECK(RunTest("l2v4", 890)); }
TEST(891) { CHECK(RunTest("l2v4", 891)); }
TEST(892) { CHECK(RunTest("l2v4", 892)); }
TEST(893) { CHECK(RunTest("l2v4", 893)); }
TEST(894) { CHECK(RunTest("l2v4", 894)); }
TEST(895) { CHECK(RunTest("l2v4", 895)); }
TEST(896) { CHECK(RunTest("l2v4", 896)); }
TEST(897) { CHECK(RunTest("l2v4", 897)); }
TEST(898) { CHECK(RunTest("l2v4", 898)); }
TEST(899) { CHECK(RunTest("l2v4", 899)); }
TEST(900) { CHECK(RunTest("l2v4", 900)); }
TEST(901) { CHECK(RunTest("l2v4", 901)); }
TEST(902) { CHECK(RunTest("l2v4", 902)); }
TEST(903) { CHECK(RunTest("l2v4", 903)); }
TEST(904) { CHECK(RunTest("l2v4", 904)); }
TEST(905) { CHECK(RunTest("l2v4", 905)); }
TEST(906) { CHECK(RunTest("l2v4", 906)); }
TEST(907) { CHECK(RunTest("l2v4", 907)); }
TEST(908) { CHECK(RunTest("l2v4", 908)); }
TEST(909) { CHECK(RunTest("l2v4", 909)); }
TEST(910) { CHECK(RunTest("l2v4", 910)); }
TEST(911) { CHECK(RunTest("l2v4", 911)); }
TEST(912) { CHECK(RunTest("l2v4", 912)); }
TEST(913) { CHECK(RunTest("l2v4", 913)); }
TEST(914) { CHECK(RunTest("l2v4", 914)); }
TEST(915) { CHECK(RunTest("l2v4", 915)); }
TEST(916) { CHECK(RunTest("l2v4", 916)); }
TEST(917) { CHECK(RunTest("l2v4", 917)); }
TEST(918) { CHECK(RunTest("l2v4", 918)); }
TEST(919) { CHECK(RunTest("l2v4", 919)); }
TEST(920) { CHECK(RunTest("l2v4", 920)); }
TEST(921) { CHECK(RunTest("l2v4", 921)); }
TEST(922) { CHECK(RunTest("l2v4", 922)); }
TEST(923) { CHECK(RunTest("l2v4", 923)); }
TEST(924) { CHECK(RunTest("l2v4", 924)); }
TEST(925) { CHECK(RunTest("l2v4", 925)); }
TEST(926) { CHECK(RunTest("l2v4", 926)); }
TEST(927) { CHECK(RunTest("l2v4", 927)); }
TEST(928) { CHECK(RunTest("l3v1", 928)); }
TEST(929) { CHECK(RunTest("l3v1", 929)); }
TEST(930) { CHECK(RunTest("l3v1", 930)); }
TEST(931) { CHECK(RunTest("l3v1", 931)); }
TEST(932) { CHECK(RunTest("l3v1", 932)); }
TEST(933) { CHECK(RunTest("l3v1", 933)); }
TEST(934) { CHECK(RunTest("l3v1", 934)); }
TEST(935) { CHECK(RunTest("l3v1", 935)); }
TEST(936) { CHECK(RunTest("l2v4", 936)); }
//TEST(937) { CHECK(RunTest("l2v4", 937)); } // CSymbolDelay
//TEST(938) { CHECK(RunTest("l2v4", 938)); } // CSymbolDelay
//TEST(939) { CHECK(RunTest("l2v4", 939)); } // CSymbolDelay
//TEST(940) { CHECK(RunTest("l2v4", 940)); } // CSymbolDelay
//TEST(941) { CHECK(RunTest("l2v4", 941)); } // CSymbolDelay
//TEST(942) { CHECK(RunTest("l2v4", 942)); } // CSymbolDelay
//TEST(943) { CHECK(RunTest("l2v4", 943)); } // CSymbolDelay
TEST(944) { CHECK(RunTest("l2v4", 944)); }
TEST(945) { CHECK(RunTest("l2v4", 945)); }
TEST(946) { CHECK(RunTest("l2v4", 946)); }
TEST(947) { CHECK(RunTest("l2v4", 947)); }
TEST(948) { CHECK(RunTest("l2v4", 948)); }
TEST(949) { CHECK(RunTest("l2v4", 949)); }
TEST(950) { CHECK(RunTest("l2v4", 950)); }
TEST(951) { CHECK(RunTest("l2v4", 951)); }
TEST(952) { CHECK(RunTest("l3v1", 952)); }
TEST(953) { CHECK(RunTest("l3v1", 953)); }
TEST(954) { CHECK(RunTest("l2v4", 954)); }
TEST(955) { CHECK(RunTest("l2v4", 955)); }
TEST(956) { CHECK(RunTest("l2v4", 956)); }
TEST(957) { CHECK(RunTest("l2v4", 957)); }
TEST(958) { CHECK(RunTest("l2v4", 958)); }
TEST(959) { CHECK(RunTest("l2v4", 959)); }
TEST(960) { CHECK(RunTest("l3v1", 960)); }
TEST(961) { CHECK(RunTest("l3v1", 961)); }
TEST(962) { CHECK(RunTest("l3v1", 962)); }
TEST(963) { CHECK(RunTest("l3v1", 963)); }
TEST(964) { CHECK(RunTest("l3v1", 964)); }
TEST(965) { CHECK(RunTest("l3v1", 965)); }
TEST(966) { CHECK(RunTest("l3v1", 966)); }
TEST(967) { CHECK(RunTest("l3v1", 967)); }
TEST(968) { CHECK(RunTest("l2v4", 968)); }
TEST(969) { CHECK(RunTest("l3v1", 969)); }
TEST(970) { CHECK(RunTest("l3v1", 970)); }
TEST(971) { CHECK(RunTest("l3v1", 971)); }
TEST(972) { CHECK(RunTest("l3v1", 972)); }
TEST(973) { CHECK(RunTest("l2v4", 973)); }
TEST(974) { CHECK(RunTest("l3v1", 974)); }
TEST(975) { CHECK(RunTest("l3v1", 975)); }
TEST(976) { CHECK(RunTest("l3v1", 976)); }
TEST(977) { CHECK(RunTest("l3v1", 977)); }
#ifdef BUILD_LLVM
TEST(978) { CHECK(RunTest("l3v1", 978)); } //Crash in legacy C mode.
#endif
TEST(979) { CHECK(RunTest("l2v4", 979)); }
TEST(980) { CHECK(RunTest("l3v1", 980)); }
//TEST(981)  { CHECK(RunTest("l3v1", 981  )); } // CSymbolDelay
//TEST(982)  { CHECK(RunTest("l3v1", 982  )); } // CSymbolDelay
//TEST(983)  { CHECK(RunTest("l3v1", 983  )); } // AlgebraicRule // CSymbolDelay
//TEST(984)  { CHECK(RunTest("l3v1", 984  )); } // CSymbolDelay
//TEST(985)  { CHECK(RunTest("l3v1", 985  )); } // CSymbolDelay
//TEST(986)  { CHECK(RunTest("l3v1", 986  )); } // FastReaction
//TEST(987)  { CHECK(RunTest("l3v1", 987  )); } // FastReaction
//TEST(988)  { CHECK(RunTest("l3v1", 988  )); } // FastReaction
TEST(989)  { CHECK(RunTest("l2v4", 989  )); }
TEST(990)  { CHECK(RunTest("l2v4", 990  )); }
TEST(991)  { CHECK(RunTest("l2v4", 991  )); }
TEST(992)  { CHECK(RunTest("l2v4", 992  )); }
//TEST(993)  { CHECK(RunTest("l2v4", 993  )); } // AlgebraicRule
TEST(994)  { CHECK(RunTest("l2v4", 994  )); }
TEST(995)  { CHECK(RunTest("l3v1", 995  )); }
TEST(996)  { CHECK(RunTest("l3v1", 996  )); }
TEST(997)  { CHECK(RunTest("l3v1", 997  )); }
TEST(998)  { CHECK(RunTest("l3v1", 998  )); }
TEST(999)  { CHECK(RunTest("l3v1", 999  )); }
TEST(1000) { CHECK(RunTest("l3v1", 1000 )); }
TEST(1001) { CHECK(RunTest("l3v1", 1001 )); }
TEST(1002) { CHECK(RunTest("l3v1", 1002 )); }
TEST(1003) { CHECK(RunTest("l3v1", 1003 )); }
TEST(1004) { CHECK(RunTest("l3v1", 1004 )); }
TEST(1005) { CHECK(RunTest("l3v1", 1005 )); }
TEST(1006) { CHECK(RunTest("l3v1", 1006 )); }
TEST(1007) { CHECK(RunTest("l3v1", 1007 )); }
TEST(1008) { CHECK(RunTest("l3v1", 1008 )); }
TEST(1009) { CHECK(RunTest("l3v1", 1009 )); }
TEST(1010) { CHECK(RunTest("l3v1", 1010 )); }
TEST(1011) { CHECK(RunTest("l3v1", 1011 )); }
TEST(1012) { CHECK(RunTest("l3v1", 1012 )); }
TEST(1013) { CHECK(RunTest("l3v1", 1013 )); }
TEST(1014) { CHECK(RunTest("l3v1", 1014 )); }
TEST(1015) { CHECK(RunTest("l3v1", 1015 )); }
TEST(1016) { CHECK(RunTest("l3v1", 1016 )); }
TEST(1017) { CHECK(RunTest("l3v1", 1017 )); }
TEST(1018) { CHECK(RunTest("l3v1", 1018 )); }
TEST(1019) { CHECK(RunTest("l3v1", 1019 )); }
TEST(1020) { CHECK(RunTest("l3v1", 1020 )); }
TEST(1021) { CHECK(RunTest("l3v1", 1021 )); }
TEST(1022) { CHECK(RunTest("l3v1", 1022 )); }
TEST(1023) { CHECK(RunTest("l3v1", 1023 )); }
TEST(1024) { CHECK(RunTest("l3v1", 1024 )); }
TEST(1025) { CHECK(RunTest("l3v1", 1025 )); }
TEST(1026) { CHECK(RunTest("l3v1", 1026 )); }
TEST(1027) { CHECK(RunTest("l2v4", 1027 )); }
TEST(1028) { CHECK(RunTest("l2v4", 1028 )); }
TEST(1029) { CHECK(RunTest("l2v4", 1029 )); }
TEST(1030) { CHECK(RunTest("l3v1", 1030 )); }
TEST(1031) { CHECK(RunTest("l3v1", 1031 )); }
TEST(1032) { CHECK(RunTest("l3v1", 1032 )); }
TEST(1033) { CHECK(RunTest("l3v1", 1033 )); }
TEST(1034) { CHECK(RunTest("l3v1", 1034 )); }
TEST(1035) { CHECK(RunTest("l3v1", 1035 )); }
TEST(1036) { CHECK(RunTest("l3v1", 1036 )); }
TEST(1037) { CHECK(RunTest("l3v1", 1037 )); }
TEST(1038) { CHECK(RunTest("l3v1", 1038 )); }
TEST(1039) { CHECK(RunTest("l3v1", 1039 )); }
TEST(1040) { CHECK(RunTest("l3v1", 1040 )); }
TEST(1041) { CHECK(RunTest("l3v1", 1041 )); }
TEST(1042) { CHECK(RunTest("l3v1", 1042 )); }
TEST(1043) { CHECK(RunTest("l3v1", 1043 )); }
//TEST(1044) { CHECK(RunTest("l3v1", 1044 )); } // AlgebraicRule
TEST(1045) { CHECK(RunTest("l3v1", 1045 )); }
TEST(1046) { CHECK(RunTest("l3v1", 1046 )); }
TEST(1047) { CHECK(RunTest("l3v1", 1047 )); }
TEST(1048) { CHECK(RunTest("l3v1", 1048 )); }
TEST(1049) { CHECK(RunTest("l3v1", 1049 )); }
TEST(1050) { CHECK(RunTest("l3v1", 1050 )); }
//TEST(1051) { CHECK(RunTest("l3v1", 1051 )); } // FastReaction
//TEST(1052) { CHECK(RunTest("l3v1", 1052 )); } // FastReaction
//TEST(1053) { CHECK(RunTest("l3v1", 1053 )); } // FastReaction
//TEST(1054) { CHECK(RunTest("l3v1", 1054 )); } // AlgebraicRule
TEST(1055) { CHECK(RunTest("l3v1", 1055 )); }
TEST(1056) { CHECK(RunTest("l3v1", 1056 )); }
TEST(1057) { CHECK(RunTest("l3v1", 1057 )); }
TEST(1058) { CHECK(RunTest("l3v1", 1058 )); }
TEST(1059) { CHECK(RunTest("l3v1", 1059 )); }
TEST(1060) { CHECK(RunTest("l3v1", 1060 )); }
TEST(1061) { CHECK(RunTest("l3v1", 1061 )); }
TEST(1062) { CHECK(RunTest("l3v1", 1062 )); }
TEST(1063) { CHECK(RunTest("l3v1", 1063 )); }
TEST(1064) { CHECK(RunTest("l3v1", 1064 )); }
TEST(1065) { CHECK(RunTest("l3v1", 1065 )); }
TEST(1066) { CHECK(RunTest("l3v1", 1066 )); }
TEST(1067) { CHECK(RunTest("l3v1", 1067 )); }
TEST(1068) { CHECK(RunTest("l3v1", 1068 )); }
TEST(1069) { CHECK(RunTest("l3v1", 1069 )); }
TEST(1070) { CHECK(RunTest("l3v1", 1070 )); }
TEST(1071) { CHECK(RunTest("l3v1", 1071 )); }
TEST(1072) { CHECK(RunTest("l3v1", 1072 )); }
TEST(1073) { CHECK(RunTest("l3v1", 1073 )); }
TEST(1074) { CHECK(RunTest("l3v1", 1074 )); }
TEST(1075) { CHECK(RunTest("l3v1", 1075 )); }
TEST(1076) { CHECK(RunTest("l3v1", 1076 )); }
TEST(1077) { CHECK(RunTest("l3v1", 1077 )); }
TEST(1078) { CHECK(RunTest("l3v1", 1078 )); }
TEST(1079) { CHECK(RunTest("l3v1", 1079 )); }
TEST(1080) { CHECK(RunTest("l3v1", 1080 )); }
TEST(1081) { CHECK(RunTest("l3v1", 1081 )); }
TEST(1082) { CHECK(RunTest("l3v1", 1082 )); }
//TEST(1083) { CHECK(RunTest("l3v1", 1083 )); } // AlgebraicRule
//TEST(1084) { CHECK(RunTest("l3v1", 1084 )); } // AlgebraicRule
//TEST(1085) { CHECK(RunTest("l3v1", 1085 )); } // AlgebraicRule
//TEST(1086) { CHECK(RunTest("l3v1", 1086 )); } // AlgebraicRule
TEST(1087) { CHECK(RunTest("l3v1", 1087 )); }
TEST(1088) { CHECK(RunTest("l3v1", 1088 )); }
TEST(1089) { CHECK(RunTest("l3v1", 1089 )); }
TEST(1090) { CHECK(RunTest("l3v1", 1090 )); }
TEST(1091) { CHECK(RunTest("l3v1", 1091 )); }
TEST(1092) { CHECK(RunTest("l3v1", 1092 )); }
TEST(1093) { CHECK(RunTest("l3v1", 1093 )); }
TEST(1094) { CHECK(RunTest("l3v1", 1094 )); }
TEST(1095) { CHECK(RunTest("l3v1", 1095 )); }
TEST(1096) { CHECK(RunTest("l3v1", 1096 )); }
TEST(1097) { CHECK(RunTest("l3v1", 1097 )); }
TEST(1098) { CHECK(RunTest("l3v1", 1098 )); }
TEST(1099) { CHECK(RunTest("l3v1", 1099 )); }
TEST(1100) { CHECK(RunTest("l3v1", 1100 )); }
TEST(1101) { CHECK(RunTest("l3v1", 1101 )); }
TEST(1102) { CHECK(RunTest("l3v1", 1102 )); }
TEST(1103) { CHECK(RunTest("l3v1", 1103 )); }
TEST(1104) { CHECK(RunTest("l3v1", 1104 )); }
TEST(1105) { CHECK(RunTest("l3v1", 1105 )); }
TEST(1106) { CHECK(RunTest("l3v1", 1106 )); }
TEST(1107) { CHECK(RunTest("l3v1", 1107 )); }
//TEST(1108) { CHECK(RunTest("l3v1", 1108 )); } // AlgebraicRule
TEST(1109) { CHECK(RunTest("l3v1", 1109 )); }
TEST(1110) { CHECK(RunTest("l3v1", 1110 )); }
TEST(1111) { CHECK(RunTest("l3v1", 1111 )); }
TEST(1112) { CHECK(RunTest("l3v1", 1112 )); }
TEST(1113) { CHECK(RunTest("l3v1", 1113 )); }
TEST(1114) { CHECK(RunTest("l3v1", 1114 )); }
TEST(1115) { CHECK(RunTest("l3v1", 1115 )); }
TEST(1116) { CHECK(RunTest("l3v1", 1116 )); }
TEST(1117) { CHECK(RunTest("l3v1", 1117 )); }
TEST(1118) { CHECK(RunTest("l3v1", 1118 )); }
TEST(1119) { CHECK(RunTest("l3v1", 1119 )); }
*/TEST(1120) { CHECK(RunTest("l3v1", 1120 )); }/*
TEST(1121) { CHECK(RunTest("l3v1", 1121 )); }
TEST(1122) { CHECK(RunTest("l3v1", 1122 )); }
TEST(1123) { CHECK(RunTest("l3v1", 1123 )); }*/
/*
 * Tests 1124 -> 1183 use the hierarchical composition extension.
 * They are included in the SBML_TEST_SUITE_COMP set, below.
 */
//TEST(1184) { CHECK(RunTest("l3v1", 1184 )); }
//TEST(1185) { CHECK(RunTest("l3v1", 1185 )); }
/*
 * Tests 1186 -> 1196 use the FBC extension.
 * They are included in the SBML_TEST_SUITE_FBC set, below.
 */

 /*
 * Tests 1197 -> 1218 are 3.2.0 tests
 */

//// ???
////TEST(1197) { CHECK(RunTest("l3v1", 1197)); }
////TEST(1198) { CHECK(RunTest("l3v1", 1198)); } //temp
//
//// Piecewise
////TEST(1199) { CHECK(RunTest("l3v1", 1199)); }
////TEST(1200) { CHECK(RunTest("l3v1", 1200)); }
////TEST(1201) { CHECK(RunTest("l3v1", 1201)); }
////TEST(1202) { CHECK(RunTest("l3v1", 1202)); }
////TEST(1203) { CHECK(RunTest("l3v1", 1203)); }
////TEST(1204) { CHECK(RunTest("l3v1", 1204)); }
//
//// Variable-size compartments
////TEST(1205) { CHECK(RunTest("l3v1", 1205)); }
////TEST(1206) { CHECK(RunTest("l3v1", 1206)); }
////TEST(1207) { CHECK(RunTest("l3v1", 1207)); }
////TEST(1208) { CHECK(RunTest("l3v1", 1208)); } //temp
//
//// MathML relational operators
////TEST(1209) { CHECK(RunTest("l3v1", 1209)); } //pass
////TEST(1210) { CHECK(RunTest("l3v1", 1210)); } //pass
////TEST(1211) { CHECK(RunTest("l3v1", 1211)); } //pass
//TEST(1212) { CHECK(RunTest("l3v1", 1212)); }
//TEST(1213) { CHECK(RunTest("l3v1", 1213)); }
////TEST(1214) { CHECK(RunTest("l3v1", 1214)); } //pass
////TEST(1215) { CHECK(RunTest("l3v1", 1215)); } //pass
//TEST(1216) { CHECK(RunTest("l3v1", 1216)); }
//
////Interdependent initial assignments and assignment rules
////TEST(1217) { CHECK(RunTest("l3v1", 1217)); }
////TEST(1218) { CHECK(RunTest("l3v1", 1218)); }
//
///*
//* Tests 1219 -> 1778 are 3.3.0 tests
//*/
//// compartment sizes other than 1
////TEST(1219) { CHECK(RunTest("l3v2", 1219)); }
////TEST(1220) { CHECK(RunTest("l3v2", 1220)); }
////TEST(1221) { CHECK(RunTest("l3v2", 1221)); }
////TEST(1222) { CHECK(RunTest("l3v2", 1222)); }
////TEST(1223) { CHECK(RunTest("l3v2", 1223)); }
//
//// reaction identifier in the MathML
////TEST(1224) { CHECK(RunTest("l3v2", 1224)); }
////TEST(1225) { CHECK(RunTest("l3v2", 1225)); }
////TEST(1226) { CHECK(RunTest("l3v2", 1226)); }
////TEST(1227) { CHECK(RunTest("l3v2", 1227)); }
////TEST(1228) { CHECK(RunTest("l3v2", 1228)); }
////TEST(1229) { CHECK(RunTest("l3v2", 1229)); }
////TEST(1230) { CHECK(RunTest("l3v2", 1230)); }
////TEST(1231) { CHECK(RunTest("l3v2", 1231)); }
////TEST(1232) { CHECK(RunTest("l3v2", 1232)); }
////TEST(1233) { CHECK(RunTest("l3v2", 1233)); }
//
//// elements without child elements
//TEST(1234) { CHECK(RunTest("l3v2", 1234)); }
//TEST(1235) { CHECK(RunTest("l3v2", 1235)); }
//TEST(1236) { CHECK(RunTest("l3v2", 1236)); }
//TEST(1237) { CHECK(RunTest("l3v2", 1237)); }
////TEST(1238) { CHECK(RunTest("l3v2", 1238)); } //temp
////TEST(1239) { CHECK(RunTest("l3v2", 1239)); } //temp
////TEST(1240) { CHECK(RunTest("l3v2", 1240)); } //pass
//TEST(1241) { CHECK(RunTest("l3v2", 1241)); }
//TEST(1242) { CHECK(RunTest("l3v2", 1242)); }
//TEST(1243) { CHECK(RunTest("l3v2", 1243)); }
////TEST(1244) { CHECK(RunTest("l3v2", 1244)); } //pass
////TEST(1245) { CHECK(RunTest("l3v2", 1245)); } //pass
////TEST(1246) { CHECK(RunTest("l3v2", 1246)); } //pass
////TEST(1247) { CHECK(RunTest("l3v2", 1247)); } //pass
//
//// rateOf
//TEST(1248) { CHECK(RunTest("l3v2", 1248)); }
//TEST(1249) { CHECK(RunTest("l3v2", 1249)); }
//TEST(1250) { CHECK(RunTest("l3v2", 1250)); }
//TEST(1251) { CHECK(RunTest("l3v2", 1251)); }
//TEST(1252) { CHECK(RunTest("l3v2", 1252)); }
//TEST(1253) { CHECK(RunTest("l3v2", 1253)); }
//TEST(1254) { CHECK(RunTest("l3v2", 1254)); }
//TEST(1255) { CHECK(RunTest("l3v2", 1255)); }
//TEST(1256) { CHECK(RunTest("l3v2", 1256)); }
////TEST(1257) { CHECK(RunTest("l3v2", 1257)); } //temp
//TEST(1258) { CHECK(RunTest("l3v2", 1258)); }
//TEST(1259) { CHECK(RunTest("l3v2", 1259)); }
//TEST(1260) { CHECK(RunTest("l3v2", 1260)); }
//TEST(1261) { CHECK(RunTest("l3v2", 1261)); }
//TEST(1262) { CHECK(RunTest("l3v2", 1262)); }
//TEST(1263) { CHECK(RunTest("l3v2", 1263)); }
//TEST(1264) { CHECK(RunTest("l3v2", 1264)); }
//TEST(1265) { CHECK(RunTest("l3v2", 1265)); }
//TEST(1266) { CHECK(RunTest("l3v2", 1266)); }
//TEST(1267) { CHECK(RunTest("l3v2", 1267)); }
////TEST(1268) { CHECK(RunTest("l3v2", 1268)); } //temp
//TEST(1269) { CHECK(RunTest("l3v2", 1269)); }
//TEST(1270) { CHECK(RunTest("l3v2", 1270)); }
//
//// elements without child elements
////TEST(1271) { CHECK(RunTest("l3v2", 1271)); } //pass
//
////max, min, rem, quotient, and implies
//TEST(1272) { CHECK(RunTest("l3v2", 1272)); }
//TEST(1273) { CHECK(RunTest("l3v2", 1273)); }
//TEST(1274) { CHECK(RunTest("l3v2", 1274)); }
//TEST(1275) { CHECK(RunTest("l3v2", 1275)); }
//TEST(1276) { CHECK(RunTest("l3v2", 1276)); }
//TEST(1277) { CHECK(RunTest("l3v2", 1277)); }
//TEST(1278) { CHECK(RunTest("l3v2", 1278)); }
//TEST(1279) { CHECK(RunTest("l3v2", 1279)); }
//TEST(1280) { CHECK(RunTest("l3v2", 1280)); }
//TEST(1281) { CHECK(RunTest("l3v2", 1281)); }
//
//// Boolean values in numeric contexts
//TEST(1282) { CHECK(RunTest("l3v2", 1282)); }
//TEST(1283) { CHECK(RunTest("l3v2", 1283)); }
//TEST(1284) { CHECK(RunTest("l3v2", 1284)); }
//TEST(1285) { CHECK(RunTest("l3v2", 1285)); }
//TEST(1286) { CHECK(RunTest("l3v2", 1286)); }
//TEST(1287) { CHECK(RunTest("l3v2", 1287)); }
//TEST(1288) { CHECK(RunTest("l3v2", 1288)); }
//TEST(1289) { CHECK(RunTest("l3v2", 1289)); }
//TEST(1290) { CHECK(RunTest("l3v2", 1290)); }
//TEST(1291) { CHECK(RunTest("l3v2", 1291)); }
//TEST(1292) { CHECK(RunTest("l3v2", 1292)); }
//
////rateOf
////TEST(1293) { CHECK(RunTest("l3v2", 1293)); } //temp
//TEST(1294) { CHECK(RunTest("l3v2", 1294)); }
//TEST(1295) { CHECK(RunTest("l3v2", 1295)); }
//TEST(1296) { CHECK(RunTest("l3v2", 1296)); }
//TEST(1297) { CHECK(RunTest("l3v2", 1297)); }
//TEST(1298) { CHECK(RunTest("l3v2", 1298)); }
////TEST(1299) { CHECK(RunTest("l3v2", 1299)); } //temp
//
//// reaction identifier in the MathML and elements without child elements
////TEST(1300) { CHECK(RunTest("l3v2", 1300)); }
////TEST(1301) { CHECK(RunTest("l3v2", 1301)); }
////TEST(1302) { CHECK(RunTest("l3v2", 1302)); }
////TEST(1303) { CHECK(RunTest("l3v2", 1303)); }
////TEST(1304) { CHECK(RunTest("l3v2", 1304)); }
////TEST(1305) { CHECK(RunTest("l3v2", 1305)); }
////TEST(1306) { CHECK(RunTest("l3v2", 1306)); }
//
//// species in the model is used as an amount
////TEST(1307) { CHECK(RunTest("l3v2", 1307)); }
////TEST(1308) { CHECK(RunTest("l3v2", 1308)); }
////TEST(1309) { CHECK(RunTest("l3v2", 1309)); }
//
//// compartment with a spatialDimensions value of 2.7
////TEST(1310) { CHECK(RunTest("l3v2", 1310)); }
//
//// nested function definitions
////TEST(1311) { CHECK(RunTest("l3v2", 1311)); }
////TEST(1312) { CHECK(RunTest("l3v2", 1312)); }
////TEST(1313) { CHECK(RunTest("l3v2", 1313)); }
////TEST(1314) { CHECK(RunTest("l3v2", 1314)); }
////TEST(1315) { CHECK(RunTest("l3v2", 1315)); }
//
//// csymbols with names that shadow other parameters or functions
////TEST(1316) { CHECK(RunTest("l3v2", 1316)); } //pass
////TEST(1317) { CHECK(RunTest("l3v2", 1317)); } //pass
//TEST(1318) { CHECK(RunTest("l3v2", 1318)); }
//TEST(1319) { CHECK(RunTest("l3v2", 1319)); }
//TEST(1320) { CHECK(RunTest("l3v2", 1320)); }
//TEST(1321) { CHECK(RunTest("l3v2", 1321)); }
//TEST(1322) { CHECK(RunTest("l3v2", 1322)); }
//
//// proper value of Avogadro's constant
////TEST(1323) { CHECK(RunTest("l3v2", 1323)); }
//
////various event semantics
////TEST(1324) { CHECK(RunTest("l3v2", 1324)); }
////TEST(1325) { CHECK(RunTest("l3v2", 1325)); }
////TEST(1326) { CHECK(RunTest("l3v2", 1326)); }
////TEST(1327) { CHECK(RunTest("l3v2", 1327)); }
////TEST(1328) { CHECK(RunTest("l3v2", 1328)); }
////TEST(1329) { CHECK(RunTest("l3v2", 1329)); }
////TEST(1330) { CHECK(RunTest("l3v2", 1330)); }
////TEST(1331) { CHECK(RunTest("l3v2", 1331)); }
////TEST(1332) { CHECK(RunTest("l3v2", 1332)); }
//TEST(1333) { CHECK(RunTest("l3v2", 1333)); }
////TEST(1334) { CHECK(RunTest("l3v2", 1334)); }
////TEST(1335) { CHECK(RunTest("l3v2", 1335)); }
////TEST(1336) { CHECK(RunTest("l3v2", 1336)); }
////TEST(1337) { CHECK(RunTest("l3v2", 1337)); }
////TEST(1338) { CHECK(RunTest("l3v2", 1338)); }
//
////rate changes between being negative and being positive
////TEST(1339) { CHECK(RunTest("l3v2", 1339)); }
////TEST(1340) { CHECK(RunTest("l3v2", 1340)); }
//
////existence of a compartment, species, and nothing else
////TEST(1341) { CHECK(RunTest("l3v2", 1341)); }
////TEST(1342) { CHECK(RunTest("l3v2", 1342)); }
//
////tanh 
////TEST(1343) { CHECK(RunTest("l3v2", 1343)); }
//
////hierarchical models
////TEST(1344) { CHECK(RunTest("l3v2", 1344)); }
////TEST(1345) { CHECK(RunTest("l3v2", 1345)); }
////TEST(1346) { CHECK(RunTest("l3v2", 1346)); }
////TEST(1347) { CHECK(RunTest("l3v2", 1347)); }
////TEST(1348) { CHECK(RunTest("l3v2", 1348)); }
////TEST(1349) { CHECK(RunTest("l3v2", 1349)); }
//////TEST(1350) { CHECK(RunTest("l3v2", 1350)); } // AlgebraicRule
////TEST(1351) { CHECK(RunTest("l3v2", 1351)); }
////TEST(1352) { CHECK(RunTest("l3v2", 1352)); }
////TEST(1353) { CHECK(RunTest("l3v2", 1353)); }
////TEST(1354) { CHECK(RunTest("l3v2", 1354)); }
////TEST(1355) { CHECK(RunTest("l3v2", 1355)); }
////TEST(1356) { CHECK(RunTest("l3v2", 1356)); }
////TEST(1357) { CHECK(RunTest("l3v2", 1357)); }
////TEST(1358) { CHECK(RunTest("l3v2", 1358)); }
//////TEST(1359) { CHECK(RunTest("l3v2", 1359)); } // AlgebraicRule
////TEST(1360) { CHECK(RunTest("l3v2", 1360)); }
////TEST(1361) { CHECK(RunTest("l3v2", 1361)); }
////TEST(1362) { CHECK(RunTest("l3v2", 1362)); }
////TEST(1363) { CHECK(RunTest("l3v2", 1363)); }
////TEST(1364) { CHECK(RunTest("l3v2", 1364)); }
////TEST(1365) { CHECK(RunTest("l3v2", 1365)); }
////TEST(1366) { CHECK(RunTest("l3v2", 1366)); }
////TEST(1367) { CHECK(RunTest("l3v2", 1367)); }
//////TEST(1368) { CHECK(RunTest("l3v2", 1368)); } // AlgebraicRule
////TEST(1369) { CHECK(RunTest("l3v2", 1369)); }
////TEST(1370) { CHECK(RunTest("l3v2", 1370)); }
////TEST(1371) { CHECK(RunTest("l3v2", 1371)); }
////TEST(1372) { CHECK(RunTest("l3v2", 1372)); }
////TEST(1373) { CHECK(RunTest("l3v2", 1373)); }
////TEST(1374) { CHECK(RunTest("l3v2", 1374)); }
////TEST(1375) { CHECK(RunTest("l3v2", 1375)); }
////TEST(1376) { CHECK(RunTest("l3v2", 1376)); }
//////TEST(1377) { CHECK(RunTest("l3v2", 1377)); } // AlgebraicRule
////TEST(1378) { CHECK(RunTest("l3v2", 1378)); }
////TEST(1379) { CHECK(RunTest("l3v2", 1379)); }
////TEST(1380) { CHECK(RunTest("l3v2", 1380)); }
////TEST(1381) { CHECK(RunTest("l3v2", 1381)); }
////TEST(1382) { CHECK(RunTest("l3v2", 1382)); }
////TEST(1383) { CHECK(RunTest("l3v2", 1383)); }
////TEST(1384) { CHECK(RunTest("l3v2", 1384)); }
////TEST(1385) { CHECK(RunTest("l3v2", 1385)); }
//////TEST(1386) { CHECK(RunTest("l3v2", 1386)); } // AlgebraicRule
////TEST(1387) { CHECK(RunTest("l3v2", 1387)); }
////TEST(1388) { CHECK(RunTest("l3v2", 1388)); }
////TEST(1389) { CHECK(RunTest("l3v2", 1389)); }
////TEST(1390) { CHECK(RunTest("l3v2", 1390)); }
////TEST(1391) { CHECK(RunTest("l3v2", 1391)); }
////TEST(1392) { CHECK(RunTest("l3v2", 1392)); }
////TEST(1393) { CHECK(RunTest("l3v2", 1393)); }
////TEST(1394) { CHECK(RunTest("l3v2", 1394)); }
//
////huge model
////TEST(1395) { CHECK(RunTest("l3v2", 1395)); }
//
////combinations of fast reactions
//////TEST(1396) { CHECK(RunTest("l3v1", 1396)); } // FastReaction
//////TEST(1397) { CHECK(RunTest("l2v4", 1397)); } // FastReaction
//////TEST(1398) { CHECK(RunTest("l3v1", 1398)); } // FastReaction
//////TEST(1399) { CHECK(RunTest("l3v1", 1399)); } // FastReaction
//
////rateOf and delay csymbols
//////TEST(1400) { CHECK(RunTest("l3v2", 1400)); } // CSymbolDelay
//////TEST(1401) { CHECK(RunTest("l3v2", 1401)); } // CSymbolDelay
//////TEST(1402) { CHECK(RunTest("l3v2", 1402)); } // CSymbolDelay
//////TEST(1403) { CHECK(RunTest("l3v2", 1403)); } // CSymbolDelay
//////TEST(1404) { CHECK(RunTest("l3v2", 1404)); } // CSymbolDelay
//////TEST(1405) { CHECK(RunTest("l3v2", 1405)); } // CSymbolDelay
//////TEST(1406) { CHECK(RunTest("l3v2", 1406)); } // CSymbolDelay
//////TEST(1407) { CHECK(RunTest("l3v2", 1407)); } // CSymbolDelay
//////TEST(1408) { CHECK(RunTest("l3v2", 1408)); } // CSymbolDelay
//////TEST(1409) { CHECK(RunTest("l3v2", 1409)); } // CSymbolDelay
//////TEST(1410) { CHECK(RunTest("l3v2", 1410)); } // CSymbolDelay
//////TEST(1411) { CHECK(RunTest("l3v2", 1411)); } // CSymbolDelay
//////TEST(1412) { CHECK(RunTest("l3v2", 1412)); } // CSymbolDelay
//////TEST(1413) { CHECK(RunTest("l3v2", 1413)); } // CSymbolDelay
//////TEST(1414) { CHECK(RunTest("l3v2", 1414)); } // CSymbolDelay
//////TEST(1415) { CHECK(RunTest("l3v2", 1415)); } // CSymbolDelay
//////TEST(1416) { CHECK(RunTest("l3v2", 1416)); } // CSymbolDelay
//////TEST(1417) { CHECK(RunTest("l3v2", 1417)); } // CSymbolDelay
//////TEST(1418) { CHECK(RunTest("l3v2", 1418)); } // CSymbolDelay
//////TEST(1419) { CHECK(RunTest("l3v2", 1419)); } // CSymbolDelay
//
////single reactions that have multiple speciesReferences
////TEST(1420) { CHECK(RunTest("l3v2", 1420)); }
////TEST(1421) { CHECK(RunTest("l3v2", 1421)); }
////TEST(1422) { CHECK(RunTest("l3v2", 1422)); }
////TEST(1423) { CHECK(RunTest("l3v2", 1423)); }
////TEST(1424) { CHECK(RunTest("l3v2", 1424)); }
////TEST(1425) { CHECK(RunTest("l3v2", 1425)); }
////TEST(1426) { CHECK(RunTest("l3v2", 1426)); }
////TEST(1427) { CHECK(RunTest("l3v2", 1427)); }
////TEST(1428) { CHECK(RunTest("l3v2", 1428)); }
////TEST(1429) { CHECK(RunTest("l3v2", 1429)); }
////TEST(1430) { CHECK(RunTest("l3v2", 1430)); }
////TEST(1431) { CHECK(RunTest("l3v2", 1431)); }
////TEST(1432) { CHECK(RunTest("l3v2", 1432)); }
////TEST(1433) { CHECK(RunTest("l3v2", 1433)); }
////TEST(1434) { CHECK(RunTest("l3v2", 1434)); }
////TEST(1435) { CHECK(RunTest("l2v4", 1435)); }
////TEST(1436) { CHECK(RunTest("l3v2", 1436)); }
////TEST(1437) { CHECK(RunTest("l2v4", 1437)); }
////TEST(1438) { CHECK(RunTest("l3v2", 1438)); }
////TEST(1439) { CHECK(RunTest("l2v4", 1439)); }
////TEST(1440) { CHECK(RunTest("l3v2", 1440)); }
////TEST(1441) { CHECK(RunTest("l2v4", 1441)); }
////TEST(1442) { CHECK(RunTest("l3v2", 1442)); }
////TEST(1443) { CHECK(RunTest("l2v4", 1443)); }
//TEST(1444) { CHECK(RunTest("l3v2", 1444)); }
//TEST(1445) { CHECK(RunTest("l3v2", 1445)); }
//TEST(1446) { CHECK(RunTest("l3v2", 1446)); }
//TEST(1447) { CHECK(RunTest("l3v2", 1447)); }
//TEST(1448) { CHECK(RunTest("l3v2", 1448)); }
////TEST(1449) { CHECK(RunTest("l3v2", 1449)); } - temp
////TEST(1450) { CHECK(RunTest("l3v2", 1450)); } - temp
////TEST(1451) { CHECK(RunTest("l3v2", 1451)); } - temp
////TEST(1452) { CHECK(RunTest("l3v2", 1452)); } - temp
////TEST(1453) { CHECK(RunTest("l3v2", 1453)); } - temp
//
////delay csymbol and the VolumeConcentrationRates
//////TEST(1454) { CHECK(RunTest("l3v2", 1454)); }  // CSymbolDelay
//
////rateOf csymbol with compartments having volumes other than 1
//TEST(1455) { CHECK(RunTest("l3v2", 1455)); }
//TEST(1456) { CHECK(RunTest("l3v2", 1456)); }
//TEST(1457) { CHECK(RunTest("l3v2", 1457)); }
//TEST(1458) { CHECK(RunTest("l3v2", 1458)); }
//TEST(1459) { CHECK(RunTest("l3v2", 1459)); }
//TEST(1460) { CHECK(RunTest("l3v2", 1460)); }
//
////rateOf csymbol that points to a parameter
//TEST(1461) { CHECK(RunTest("l3v2", 1461)); }
//
////rateOf csymbol with VolumeConcentrationRates
//TEST(1462) { CHECK(RunTest("l3v2", 1462)); }
//TEST(1463) { CHECK(RunTest("l3v2", 1463)); }
//
////assigned stoichiometries with no MathML
//TEST(1464) { CHECK(RunTest("l3v2", 1464)); }
//TEST(1465) { CHECK(RunTest("l3v2", 1465)); }
//
////random event execution at time = 0
////TEST(1466) { CHECK(RunTest("l3v2", 1466)); }
//
////conversion factors separated by submodel indirection
////TEST(1467) { CHECK(RunTest("l3v2", 1467)); }
////TEST(1468) { CHECK(RunTest("l3v2", 1468)); }
////TEST(1469) { CHECK(RunTest("l3v2", 1469)); }
////TEST(1470) { CHECK(RunTest("l3v2", 1470)); }
//
////combinations of comp elements, including external model definitions, conversion factors, and the replacedBy construct
////TEST(1471) { CHECK(RunTest("l3v2", 1471)); }
////TEST(1472) { CHECK(RunTest("l3v2", 1472)); }
////TEST(1473) { CHECK(RunTest("l3v2", 1473)); }
////TEST(1474) { CHECK(RunTest("l3v2", 1474)); }
////TEST(1475) { CHECK(RunTest("l3v2", 1475)); }
////TEST(1476) { CHECK(RunTest("l3v2", 1476)); }
////TEST(1477) { CHECK(RunTest("l3v2", 1477)); }
//
////volume concentration rates with a function definition
////TEST(1478) { CHECK(RunTest("l3v2", 1478)); } - temp
//
////csymbol avogadro
//TEST(1479) { CHECK(RunTest("l3v2", 1479)); }
//
////stoichiometry math with the delay csymbol
//////TEST(1480) { CHECK(RunTest("l3v2", 1480)); } // CSymbolDelay
//////TEST(1481) { CHECK(RunTest("l2v4", 1481)); } // CSymbolDelay
//
////rateOf csymbol in an algebraic rule
//////TEST(1482) { CHECK(RunTest("l3v2", 1482)); } // AlgebraicRule
//////TEST(1483) { CHECK(RunTest("l3v2", 1483)); } // AlgebraicRule
//////TEST(1484) { CHECK(RunTest("l3v2", 1484)); } // AlgebraicRule
//
////combinations of uncommon MathML
////TEST(1485) { CHECK(RunTest("l3v2", 1485)); } - temp
////TEST(1486) { CHECK(RunTest("l3v2", 1486)); } - temp
////TEST(1487) { CHECK(RunTest("l3v2", 1487)); } - temp
////TEST(1488) { CHECK(RunTest("l3v2", 1488)); } - temp
////TEST(1489) { CHECK(RunTest("l3v2", 1489)); } - temp
////TEST(1490) { CHECK(RunTest("l3v2", 1490)); } - temp
////TEST(1491) { CHECK(RunTest("l3v2", 1491)); } - temp
////TEST(1492) { CHECK(RunTest("l3v2", 1492)); } - temp
////TEST(1493) { CHECK(RunTest("l3v2", 1493)); } - temp
////TEST(1494) { CHECK(RunTest("l3v2", 1494)); } - temp
////TEST(1495) { CHECK(RunTest("l3v2", 1495)); } - temp
////TEST(1496) { CHECK(RunTest("l3v2", 1496)); } - temp
////TEST(1497) { CHECK(RunTest("l3v2", 1497)); } - temp
//
////combination of species references in MathML with volume concentration rates
////TEST(1498) { CHECK(RunTest("l3v2", 1498)); } - temp
//
////algebraic rules with conversion factors and uncommon MathML
//////TEST(1499) { CHECK(RunTest("l3v2", 1499)); } // AlgebraicRule
//////TEST(1500) { CHECK(RunTest("l3v2", 1500)); } // AlgebraicRule
//////TEST(1501) { CHECK(RunTest("l3v2", 1501)); } // AlgebraicRule
//////TEST(1502) { CHECK(RunTest("l3v2", 1502)); } // AlgebraicRule
//////TEST(1503) { CHECK(RunTest("l3v2", 1503)); } // AlgebraicRule
//
////volume concentration rates in concert with events of various types
////TEST(1504) { CHECK(RunTest("l3v2", 1504)); } - temp
////TEST(1505) { CHECK(RunTest("l3v2", 1505)); } - temp
////TEST(1506) { CHECK(RunTest("l3v2", 1506)); } - temp
////TEST(1507) { CHECK(RunTest("l3v2", 1507)); } - temp
////TEST(1508) { CHECK(RunTest("l3v2", 1508)); } - temp
////TEST(1509) { CHECK(RunTest("l3v2", 1509)); } - temp
////TEST(1510) { CHECK(RunTest("l3v2", 1510)); } - temp
////TEST(1511) { CHECK(RunTest("l3v2", 1511)); } - temp
////TEST(1512) { CHECK(RunTest("l3v2", 1512)); } - temp
//
////volume concentration rates in concert with initial assignments
////TEST(1513) { CHECK(RunTest("l3v2", 1513)); } - temp
////TEST(1514) { CHECK(RunTest("l3v2", 1514)); } - temp
//
////uncommon MathML in rate rules and with assigned stoichiometries
////TEST(1515) { CHECK(RunTest("l3v2", 1515)); } - temp
////TEST(1516) { CHECK(RunTest("l3v2", 1516)); } - temp
////TEST(1517) { CHECK(RunTest("l2v4", 1517)); } - temp
//
////csymbol delay with various types of events
//////TEST(1518) { CHECK(RunTest("l3v2", 1518)); } // CSymbolDelay
//////TEST(1519) { CHECK(RunTest("l3v2", 1519)); } // CSymbolDelay
//////TEST(1520) { CHECK(RunTest("l3v2", 1520)); } // CSymbolDelay
//////TEST(1521) { CHECK(RunTest("l3v2", 1521)); } // CSymbolDelay
//////TEST(1522) { CHECK(RunTest("l3v2", 1522)); } // CSymbolDelay
//////TEST(1523) { CHECK(RunTest("l3v2", 1523)); } // CSymbolDelay
//////TEST(1524) { CHECK(RunTest("l3v2", 1524)); } // CSymbolDelay
//
////rateOf csymbol with various types of events
////TEST(1525) { CHECK(RunTest("l3v2", 1525)); } - temp
////TEST(1526) { CHECK(RunTest("l3v2", 1526)); } - temp
////TEST(1527) { CHECK(RunTest("l3v2", 1527)); } - temp
////TEST(1528) { CHECK(RunTest("l3v2", 1528)); } - temp
////TEST(1529) { CHECK(RunTest("l3v2", 1529)); } - temp
//
////uncommon MathML in event triggers, assignments, priorities, and delays
////TEST(1530) { CHECK(RunTest("l3v2", 1530)); }
////TEST(1531) { CHECK(RunTest("l3v2", 1531)); }
////TEST(1532) { CHECK(RunTest("l3v2", 1532)); }
////TEST(1533) { CHECK(RunTest("l3v2", 1533)); }
//
////csymbol delay when assigning to stoichiometries, with boundary species, and with fast reactions
//////TEST(1534) { CHECK(RunTest("l3v2", 1534)); } // CSymbolDelay
//////TEST(1535) { CHECK(RunTest("l3v2", 1535)); } // CSymbolDelay
//////TEST(1536) { CHECK(RunTest("l3v2", 1536)); } // CSymbolDelay
//////TEST(1537) { CHECK(RunTest("l3v2", 1537)); } // CSymbolDelay
//////TEST(1538) { CHECK(RunTest("l3v2", 1538)); } // CSymbolDelay
//////TEST(1539) { CHECK(RunTest("l3v1", 1539)); } // CSymbolDelay
//
////rateOf csymbol with boundary species and with assigned stoichiometries
//TEST(1540) { CHECK(RunTest("l3v2", 1540)); }
//TEST(1541) { CHECK(RunTest("l3v2", 1541)); }
//TEST(1542) { CHECK(RunTest("l3v2", 1542)); }
//TEST(1543) { CHECK(RunTest("l3v2", 1543)); }
//
////conversion factors in reactions that have a fast attribute value of true
//TEST(1544) { CHECK(RunTest("l3v1", 1544)); }
//TEST(1545) { CHECK(RunTest("l3v1", 1545)); }
//TEST(1546) { CHECK(RunTest("l3v1", 1546)); }
//
////reactions that have fast="true" with function definitions having various kinetic laws
//TEST(1547) { CHECK(RunTest("l3v1", 1547)); }
//TEST(1548) { CHECK(RunTest("l3v1", 1548)); }
//TEST(1549) { CHECK(RunTest("l3v1", 1549)); }
//TEST(1550) { CHECK(RunTest("l3v1", 1550)); }
//TEST(1551) { CHECK(RunTest("l3v1", 1551)); }
//
////assigning to stoichiometries, but with missing MathML
//TEST(1552) { CHECK(RunTest("l3v2", 1552)); }
//TEST(1553) { CHECK(RunTest("l3v2", 1553)); }
//TEST(1554) { CHECK(RunTest("l3v2", 1554)); }
//
////assigning to boundary species, but with missing MathML
////TEST(1555) { CHECK(RunTest("l3v2", 1555)); } - temp
////TEST(1556) { CHECK(RunTest("l3v2", 1556)); } - temp
////TEST(1557) { CHECK(RunTest("l3v2", 1557)); } - temp
//
////referencing a species reference from a fast reaction
//////TEST(1558) { CHECK(RunTest("l3v1", 1558)); } // FastReaction
//////TEST(1559) { CHECK(RunTest("l3v1", 1559)); } // FastReaction
//////TEST(1560) { CHECK(RunTest("l3v1", 1560)); } // FastReaction
//
////uncommon MathML used to assign to stoichiometries
////TEST(1561) { CHECK(RunTest("l3v2", 1561)); }
////TEST(1562) { CHECK(RunTest("l2v4", 1562)); }
////TEST(1563) { CHECK(RunTest("l3v2", 1563)); }
//
////uncommon MathML used to assign to kinetic laws of normal and fast reactions
//TEST(1564) { CHECK(RunTest("l3v2", 1564)); }
//////TEST(1565) { CHECK(RunTest("l3v1", 1565)); } // FastReaction
//
////uncommon MathML that use a species reference
////TEST(1566) { CHECK(RunTest("l3v2", 1566)); }
//
////fast reactions with algebraic rules, assignment rules, rate rules, and initial assignments
//////TEST(1567) { CHECK(RunTest("l3v1", 1567)); } // FastReaction // AlgebraicRule
//////TEST(1568) { CHECK(RunTest("l3v1", 1568)); } // FastReaction
//////TEST(1569) { CHECK(RunTest("l3v1", 1569)); } // FastReaction
//////TEST(1570) { CHECK(RunTest("l3v1", 1570)); } // FastReaction
//
////fast reactions with assigned stoichiometries
//////TEST(1571) { CHECK(RunTest("l3v1", 1571)); } // FastReaction
//////TEST(1572) { CHECK(RunTest("l3v1", 1572)); } // FastReaction
//
////assigned stoichiometries for boundary species
////TEST(1573) { CHECK(RunTest("l2v4", 1573)); }
////TEST(1574) { CHECK(RunTest("l3v2", 1574)); } - temp
//
////algebraic rules with various types of events and event elements
//////TEST(1575) { CHECK(RunTest("l3v2", 1575)); } // AlgebraicRule
//////TEST(1576) { CHECK(RunTest("l3v2", 1576)); } // AlgebraicRule
//////TEST(1577) { CHECK(RunTest("l3v2", 1577)); } // AlgebraicRule
//////TEST(1578) { CHECK(RunTest("l3v2", 1578)); } // AlgebraicRule
//////TEST(1579) { CHECK(RunTest("l3v2", 1579)); } // AlgebraicRule
//
////assigned stoichiometries in the context of various types of events
////TEST(1580) { CHECK(RunTest("l3v2", 1580)); }
////TEST(1581) { CHECK(RunTest("l2v4", 1581)); }
////TEST(1582) { CHECK(RunTest("l3v2", 1582)); }
////TEST(1583) { CHECK(RunTest("l3v2", 1583)); }
////TEST(1584) { CHECK(RunTest("l3v2", 1584)); }
////TEST(1585) { CHECK(RunTest("l2v4", 1585)); }
////TEST(1586) { CHECK(RunTest("l3v2", 1586)); }
////TEST(1587) { CHECK(RunTest("l2v4", 1587)); }
//
////random events with initial assignments, algebraic rules, rate rules, delayed events, and the csymbol delay
////TEST(1588) { CHECK(RunTest("l3v2", 1588)); }
//////TEST(1589) { CHECK(RunTest("l3v2", 1589)); } // AlgebraicRule
////TEST(1590) { CHECK(RunTest("l3v2", 1590)); }
////TEST(1591) { CHECK(RunTest("l3v2", 1591)); }
//////TEST(1592) { CHECK(RunTest("l3v2", 1592)); } // CSymbolDelay
//////TEST(1593) { CHECK(RunTest("l3v2", 1593)); } // CSymbolDelay
//
////uncommon MathML in various types of events
////TEST(1594) { CHECK(RunTest("l3v2", 1594)); }
////TEST(1595) { CHECK(RunTest("l3v2", 1595)); }
////TEST(1596) { CHECK(RunTest("l3v2", 1596)); }
////TEST(1597) { CHECK(RunTest("l3v2", 1597)); }
////TEST(1598) { CHECK(RunTest("l3v2", 1598)); }
////TEST(1599) { CHECK(RunTest("l3v2", 1599)); }
//
////missing MathML in delayed events
//TEST(1600) { CHECK(RunTest("l3v2", 1600)); }
//TEST(1601) { CHECK(RunTest("l3v2", 1601)); }
//TEST(1602) { CHECK(RunTest("l3v2", 1602)); }
//TEST(1603) { CHECK(RunTest("l3v2", 1603)); }
//
////time csymbol in a delayed event assignment
//// TEST(1604) { CHECK(RunTest("l3v2", 1604)); }
//
////random events with additional event assignments with no MathML
//TEST(1605) { CHECK(RunTest("l3v2", 1605)); }
//
////random events with stoichiometries and volume-concentration rates
//TEST(1626) { CHECK(RunTest("l3v2", 1626)); }
////TEST(1627) { CHECK(RunTest("l3v2", 1627)); } - temp
//
////multiple reactions with assigned stoichiometries affect the same species
////TEST(1631) { CHECK(RunTest("l3v2", 1631)); }
////TEST(1632) { CHECK(RunTest("l2v4", 1632)); }
////TEST(1633) { CHECK(RunTest("l3v2", 1633)); }
////TEST(1634) { CHECK(RunTest("l2v4", 1634)); }
//
////assigning variable stoichiometries to boundary species
////TEST(1635) { CHECK(RunTest("l3v2", 1635)); }
////TEST(1636) { CHECK(RunTest("l3v2", 1636)); }
//
////variable stoichiometry with the same ID as a local parameter
////TEST(1637) { CHECK(RunTest("l3v1", 1637)); }
//
////boundary species and local parameter ID shadowing
////TEST(1638) { CHECK(RunTest("l3v1", 1638)); }
////TEST(1639) { CHECK(RunTest("l3v1", 1639)); }
////TEST(1640) { CHECK(RunTest("l3v1", 1640)); }
//
////avogadro and time with the name of the other csymbol
////TEST(1641) { CHECK(RunTest("l3v2", 1641)); }
//
////conversion factors with function definitions, substance-only species, local parameters, and species references
////TEST(1642) { CHECK(RunTest("l3v2", 1642)); }
////TEST(1643) { CHECK(RunTest("l3v2", 1643)); }
////TEST(1644) { CHECK(RunTest("l3v2", 1644)); }
////TEST(1645) { CHECK(RunTest("l3v2", 1645)); }
////TEST(1646) { CHECK(RunTest("l3v2", 1646)); }
////TEST(1647) { CHECK(RunTest("l3v2", 1647)); }
////TEST(1648) { CHECK(RunTest("l3v2", 1648)); }
////TEST(1649) { CHECK(RunTest("l3v2", 1649)); }
////TEST(1650) { CHECK(RunTest("l3v2", 1650)); }
////TEST(1651) { CHECK(RunTest("l3v2", 1651)); }
////TEST(1652) { CHECK(RunTest("l3v2", 1652)); }
////TEST(1653) { CHECK(RunTest("l3v2", 1653)); }
//
////calling a function definition with a species reference
////TEST(1654) { CHECK(RunTest("l3v2", 1654)); }
//
////species reference to a substance-only species
////TEST(1655) { CHECK(RunTest("l3v2", 1655)); }
//
////local parameters that shadow species reference IDs, in various MathML contexts
////TEST(1656) { CHECK(RunTest("l3v1", 1656)); }
//
////species reference with an empty assignment rule
//TEST(1657) { CHECK(RunTest("l3v2", 1657)); }
//
////csymbol avogadro in various event elements and a rate rule
////TEST(1658) { CHECK(RunTest("l3v2", 1658)); }
////TEST(1659) { CHECK(RunTest("l3v2", 1659)); }
////TEST(1660) { CHECK(RunTest("l3v2", 1660)); }
////TEST(1661) { CHECK(RunTest("l3v2", 1661)); }
////TEST(1662) { CHECK(RunTest("l3v2", 1662)); }
////TEST(1663) { CHECK(RunTest("l3v2", 1663)); }
////TEST(1664) { CHECK(RunTest("l3v2", 1664)); }
////TEST(1665) { CHECK(RunTest("l3v2", 1665)); }
//
////conversion factors are set for species only affected by rate rules
////TEST(1666) { CHECK(RunTest("l3v2", 1666)); }
////TEST(1667) { CHECK(RunTest("l3v2", 1667)); }
////TEST(1668) { CHECK(RunTest("l3v2", 1668)); }
//
////conversion factors for species also affected by or used in events
////TEST(1669) { CHECK(RunTest("l3v2", 1669)); }
////TEST(1670) { CHECK(RunTest("l3v2", 1670)); }
////TEST(1671) { CHECK(RunTest("l3v2", 1671)); }
////TEST(1672) { CHECK(RunTest("l3v2", 1672)); }
////TEST(1673) { CHECK(RunTest("l3v2", 1673)); }
////TEST(1674) { CHECK(RunTest("l3v2", 1674)); }
////TEST(1675) { CHECK(RunTest("l3v2", 1675)); }
////TEST(1676) { CHECK(RunTest("l3v2", 1676)); }
////TEST(1677) { CHECK(RunTest("l3v2", 1677)); }
////TEST(1678) { CHECK(RunTest("l3v2", 1678)); }
////TEST(1679) { CHECK(RunTest("l3v2", 1679)); }
////TEST(1680) { CHECK(RunTest("l3v2", 1680)); }
////TEST(1681) { CHECK(RunTest("l3v2", 1681)); }
////TEST(1682) { CHECK(RunTest("l3v2", 1682)); }
////TEST(1683) { CHECK(RunTest("l3v2", 1683)); }
////TEST(1684) { CHECK(RunTest("l3v2", 1684)); }
////TEST(1685) { CHECK(RunTest("l3v2", 1685)); }
////TEST(1686) { CHECK(RunTest("l3v2", 1686)); }
////TEST(1687) { CHECK(RunTest("l3v2", 1687)); }
////TEST(1688) { CHECK(RunTest("l3v2", 1688)); }
////TEST(1689) { CHECK(RunTest("l3v2", 1689)); }
////TEST(1690) { CHECK(RunTest("l3v2", 1690)); }
////TEST(1691) { CHECK(RunTest("l3v2", 1691)); }
////TEST(1692) { CHECK(RunTest("l3v2", 1692)); }
//
////function definitions and various events, including ones that fire at t0, and those affected by assignment-time or trigger-time settings
////TEST(1693) { CHECK(RunTest("l3v2", 1693)); }
////TEST(1694) { CHECK(RunTest("l3v2", 1694)); }
////TEST(1695) { CHECK(RunTest("l3v2", 1695)); }
//TEST(1696) { CHECK(RunTest("l3v2", 1696)); }
//TEST(1697) { CHECK(RunTest("l3v2", 1697)); }
//TEST(1698) { CHECK(RunTest("l3v2", 1698)); }
//TEST(1699) { CHECK(RunTest("l3v2", 1699)); }
////TEST(1700) { CHECK(RunTest("l3v2", 1700)); }
////TEST(1701) { CHECK(RunTest("l3v2", 1701)); }
////TEST(1702) { CHECK(RunTest("l3v2", 1702)); }
//
////substance-only species together with various types of events
////TEST(1703) { CHECK(RunTest("l3v2", 1703)); }
////TEST(1704) { CHECK(RunTest("l3v2", 1704)); }
////TEST(1705) { CHECK(RunTest("l3v2", 1705)); }
////TEST(1706) { CHECK(RunTest("l3v2", 1706)); }
////TEST(1707) { CHECK(RunTest("l3v2", 1707)); }
////TEST(1708) { CHECK(RunTest("l3v2", 1708)); }
////TEST(1709) { CHECK(RunTest("l3v2", 1709)); }
//
////local parameters shadowing global parameters that are used in various event constructs, and for different types of events
////TEST(1710) { CHECK(RunTest("l3v2", 1710)); }
////TEST(1711) { CHECK(RunTest("l3v2", 1711)); }
////TEST(1712) { CHECK(RunTest("l3v2", 1712)); }
////TEST(1713) { CHECK(RunTest("l3v2", 1713)); }
////TEST(1714) { CHECK(RunTest("l3v2", 1714)); }
////TEST(1715) { CHECK(RunTest("l3v2", 1715)); }
////TEST(1716) { CHECK(RunTest("l3v2", 1716)); }
//
////referencing stoichiometries in various types of events
////TEST(1717) { CHECK(RunTest("l3v2", 1717)); }
////TEST(1718) { CHECK(RunTest("l3v2", 1718)); }
////TEST(1719) { CHECK(RunTest("l3v2", 1719)); }
////TEST(1720) { CHECK(RunTest("l3v2", 1720)); }
////TEST(1721) { CHECK(RunTest("l3v2", 1721)); }
//
////avogadro with assigned stoichiometries
////TEST(1722) { CHECK(RunTest("l3v2", 1722)); }
////TEST(1723) { CHECK(RunTest("l3v2", 1723)); }
//
////conversion factors, assigned stoichiometries, and boundary conditions in different combinations
////TEST(1724) { CHECK(RunTest("l3v2", 1724)); }
////TEST(1725) { CHECK(RunTest("l3v2", 1725)); }
////TEST(1726) { CHECK(RunTest("l3v2", 1726)); }
////TEST(1727) { CHECK(RunTest("l3v2", 1727)); }
////TEST(1728) { CHECK(RunTest("l3v2", 1728)); }
////TEST(1729) { CHECK(RunTest("l3v2", 1729)); }
////TEST(1730) { CHECK(RunTest("l3v2", 1730)); }
////TEST(1731) { CHECK(RunTest("l3v2", 1731)); }
////TEST(1732) { CHECK(RunTest("l3v2", 1732)); }
////TEST(1733) { CHECK(RunTest("l3v2", 1733)); }
////TEST(1734) { CHECK(RunTest("l3v2", 1734)); }
////TEST(1735) { CHECK(RunTest("l3v2", 1735)); }
////TEST(1736) { CHECK(RunTest("l3v2", 1736)); }
////TEST(1737) { CHECK(RunTest("l3v2", 1737)); }
////TEST(1738) { CHECK(RunTest("l3v2", 1738)); }
//
////conversion factors and local parameters
////TEST(1739) { CHECK(RunTest("l3v2", 1739)); }
////TEST(1740) { CHECK(RunTest("l3v2", 1740)); }
////TEST(1741) { CHECK(RunTest("l3v2", 1741)); }
//
////stoichiometries assigned with function definitions
////TEST(1742) { CHECK(RunTest("l3v2", 1742)); }
////TEST(1743) { CHECK(RunTest("l2v4", 1743)); }
////TEST(1744) { CHECK(RunTest("l3v2", 1744)); }
////TEST(1745) { CHECK(RunTest("l2v4", 1745)); }
//
////substance-only species in reactions with set stoichiometries
////TEST(1746) { CHECK(RunTest("l3v2", 1746)); }
////TEST(1747) { CHECK(RunTest("l2v4", 1747)); }
////TEST(1748) { CHECK(RunTest("l3v2", 1748)); }
////TEST(1749) { CHECK(RunTest("l2v4", 1749)); }
//
////local parameters shadowing variable stoichiometries and boundary species
////TEST(1750) { CHECK(RunTest("l3v1", 1750)); }
////TEST(1751) { CHECK(RunTest("l2v4", 1751)); }
////TEST(1752) { CHECK(RunTest("l3v1", 1752)); }
////TEST(1753) { CHECK(RunTest("l3v1", 1753)); }
//
////initial assignments with various types of events, especially those that fire at time = 0
////TEST(1754) { CHECK(RunTest("l3v2", 1754)); }
////TEST(1755) { CHECK(RunTest("l3v2", 1755)); }
////TEST(1756) { CHECK(RunTest("l3v2", 1756)); }
//TEST(1757) { CHECK(RunTest("l3v2", 1757)); }
//
//// delayed events that trigger themselves
////TEST(1758) { CHECK(RunTest("l3v2", 1758)); }
////TEST(1759) { CHECK(RunTest("l3v2", 1759)); }
//
////potential conflict between the avogadro csymbol with parameters and local parameters named avogadro
////TEST(1760) { CHECK(RunTest("l3v2", 1760)); }
////TEST(1761) { CHECK(RunTest("l3v2", 1761)); }
////TEST(1762) { CHECK(RunTest("l3v2", 1762)); }
////TEST(1763) { CHECK(RunTest("l3v2", 1763)); }
//
////local parameters that shadow species reference IDs, in various MathML contexts
////TEST(1764) { CHECK(RunTest("l3v1", 1764)); }
////TEST(1765) { CHECK(RunTest("l3v1", 1765)); }
////TEST(1766) { CHECK(RunTest("l3v2", 1766)); }
////TEST(1767) { CHECK(RunTest("l3v2", 1767)); }
////TEST(1768) { CHECK(RunTest("l3v2", 1768)); }
////TEST(1769) { CHECK(RunTest("l3v2", 1769)); }
////TEST(1770) { CHECK(RunTest("l3v2", 1770)); }
////TEST(1771) { CHECK(RunTest("l3v2", 1771)); }
////TEST(1772) { CHECK(RunTest("l3v2", 1772)); }
////TEST(1773) { CHECK(RunTest("l3v2", 1773)); }
////TEST(1774) { CHECK(RunTest("l3v2", 1774)); }
//
////avogadro to set conversion factor values
////TEST(1775) { CHECK(RunTest("l3v2", 1775)); }
////TEST(1776) { CHECK(RunTest("l3v2", 1776)); }
////TEST(1777) { CHECK(RunTest("l3v2", 1777)); }
//
////combinations of comp elements, including external model definitions, conversion factors, and the replacedBy construct
////TEST(1778) { CHECK(RunTest("l3v2", 1778)); }

}


/**
 * The original test suite used for Valgrind testing
 */
SUITE(SBML_TEST_SUITE_VG1)
{
    TEST(15)  { CHECK(RunTest("l2v4", 15)); }
    TEST(16)  { CHECK(RunTest("l2v4", 16)); }
    TEST(18)  { CHECK(RunTest("l2v4", 18)); }
    TEST(19)  { CHECK(RunTest("l2v4", 19)); }
    TEST(27)  { CHECK(RunTest("l2v4", 27)); }
    TEST(29)  { CHECK(RunTest("l2v4", 29)); }
    TEST(31)  { CHECK(RunTest("l2v4", 31)); }
    TEST(889) { CHECK(RunTest("l2v4", 889)); }
    TEST(890) { CHECK(RunTest("l2v4", 890)); }
    TEST(891) { CHECK(RunTest("l2v4", 891)); }
    TEST(892) { CHECK(RunTest("l2v4", 892)); }
    TEST(893) { CHECK(RunTest("l2v4", 893)); }
    TEST(894) { CHECK(RunTest("l2v4", 894)); }
    TEST(895) { CHECK(RunTest("l2v4", 895)); }
    TEST(896) { CHECK(RunTest("l2v4", 896)); }
    TEST(952) { CHECK(RunTest("l3v1", 952)); }
    TEST(408) { CHECK(RunTest("l2v4", 408)); }
    // TEST(1183) { CHECK(RunTest("l3v1", 1183 )); } This has comp extension,
    // which leaks like a pig so its useless for testing.
}

/**
 * another suite with more rate rules
 */
SUITE(SBML_TEST_SUITE_VG2)
{
    TEST(953) { CHECK(RunTest("l3v1", 953)); }
    //TEST(17)    { CHECK(RunTest("l2v4", 17));    }
    //TEST(32)    { CHECK(RunTest("l2v4", 32));    }
    //TEST(86)    { CHECK(RunTest("l2v4", 86));    }
    //TEST(165)   { CHECK(RunTest("l2v4", 165));   }
    //TEST(889)   { CHECK(RunTest("l2v4", 889));   }
    //TEST(1046)  { CHECK(RunTest("l3v1", 1046 )); }
}


SUITE(SBML_TEST_SUITE_RATE_RULES)
{
   TEST(31)  { CHECK(RunTest("l2v4", 31)); }
   TEST(32)  { CHECK(RunTest("l2v4", 32)); }
   TEST(33)  { CHECK(RunTest("l2v4", 33)); }
   //TEST(40)  { CHECK(RunTest("l2v4", 40)); }
   TEST(51)  { CHECK(RunTest("l2v4", 51)); }
   TEST(52)  { CHECK(RunTest("l2v4", 52)); }
   TEST(53)  { CHECK(RunTest("l2v4", 53)); }
   TEST(170)  { CHECK(RunTest("l2v4", 170)); }
   TEST(185)  { CHECK(RunTest("l2v4", 185)); }
   TEST(310)  { CHECK(RunTest("l2v4", 310)); }
   TEST(316)  { CHECK(RunTest("l2v4", 316)); }
   TEST(461)  { CHECK(RunTest("l2v4", 461)); }
   TEST(712)  { CHECK(RunTest("l2v4", 712)); }
}

SUITE(SBML_TEST_SUITE_EVENTS)
{
    TEST(26) { CHECK(RunTest("l2v4", 26)); }
    TEST(41) { CHECK(RunTest("l2v4", 41)); }
    TEST(71) { CHECK(RunTest("l2v4", 71)); }
    TEST(72) { CHECK(RunTest("l2v4", 72)); }
    TEST(73) { CHECK(RunTest("l2v4", 73)); }
    TEST(74) { CHECK(RunTest("l2v4", 74)); }

    TEST(351) { CHECK(RunTest("l2v4", 351)); }
    TEST(352) { CHECK(RunTest("l2v4", 352)); }
    TEST(353) { CHECK(RunTest("l2v4", 353)); }
    TEST(354) { CHECK(RunTest("l2v4", 354)); }
    TEST(355) { CHECK(RunTest("l2v4", 355)); }
    TEST(356) { CHECK(RunTest("l2v4", 356)); }
    TEST(357) { CHECK(RunTest("l2v4", 357)); }
    TEST(358) { CHECK(RunTest("l2v4", 358)); }
    TEST(359) { CHECK(RunTest("l2v4", 359)); }
    TEST(360) { CHECK(RunTest("l2v4", 360)); }
    TEST(361) { CHECK(RunTest("l2v4", 361)); }
    TEST(362) { CHECK(RunTest("l2v4", 362)); }
    TEST(363) { CHECK(RunTest("l2v4", 363)); }
    TEST(364) { CHECK(RunTest("l2v4", 364)); }
    TEST(365) { CHECK(RunTest("l2v4", 365)); }
    TEST(366) { CHECK(RunTest("l2v4", 366)); }
    TEST(367) { CHECK(RunTest("l2v4", 367)); }
    TEST(368) { CHECK(RunTest("l2v4", 368)); }
    TEST(369) { CHECK(RunTest("l2v4", 369)); }
    TEST(370) { CHECK(RunTest("l2v4", 370)); }
    TEST(371) { CHECK(RunTest("l2v4", 371)); }
    TEST(372) { CHECK(RunTest("l2v4", 372)); }
    TEST(373) { CHECK(RunTest("l2v4", 373)); }
    TEST(374) { CHECK(RunTest("l2v4", 374)); }
    TEST(375) { CHECK(RunTest("l2v4", 375)); }
    TEST(376) { CHECK(RunTest("l2v4", 376)); }
    TEST(377) { CHECK(RunTest("l2v4", 377)); }
    TEST(378) { CHECK(RunTest("l2v4", 378)); }
    TEST(379) { CHECK(RunTest("l2v4", 379)); }
    TEST(380) { CHECK(RunTest("l2v4", 380)); }
    TEST(381) { CHECK(RunTest("l2v4", 381)); }
    TEST(382) { CHECK(RunTest("l2v4", 382)); }
    TEST(383) { CHECK(RunTest("l2v4", 383)); }
    TEST(384) { CHECK(RunTest("l2v4", 384)); }
    TEST(385) { CHECK(RunTest("l2v4", 385)); }
    TEST(386) { CHECK(RunTest("l2v4", 386)); }
    TEST(387) { CHECK(RunTest("l2v4", 387)); }
    TEST(388) { CHECK(RunTest("l2v4", 388)); }
    TEST(389) { CHECK(RunTest("l2v4", 389)); }
    TEST(390) { CHECK(RunTest("l2v4", 390)); }
    TEST(391) { CHECK(RunTest("l2v4", 391)); }
    TEST(392) { CHECK(RunTest("l2v4", 392)); }
    TEST(393) { CHECK(RunTest("l2v4", 393)); }
    TEST(394) { CHECK(RunTest("l2v4", 394)); }
    TEST(395) { CHECK(RunTest("l2v4", 395)); }
    TEST(396) { CHECK(RunTest("l2v4", 396)); }
    TEST(397) { CHECK(RunTest("l2v4", 397)); }
    TEST(398) { CHECK(RunTest("l2v4", 398)); }
    TEST(399) { CHECK(RunTest("l2v4", 399)); }
    TEST(400) { CHECK(RunTest("l2v4", 400)); }
    TEST(401) { CHECK(RunTest("l2v4", 401)); }
    TEST(402) { CHECK(RunTest("l2v4", 402)); }
    TEST(403) { CHECK(RunTest("l2v4", 403)); }
    TEST(404) { CHECK(RunTest("l2v4", 404)); }
    TEST(405) { CHECK(RunTest("l2v4", 405)); }
    TEST(406) { CHECK(RunTest("l2v4", 406)); }
    TEST(407) { CHECK(RunTest("l2v4", 407)); }
    TEST(408) { CHECK(RunTest("l2v4", 408)); }
    TEST(409) { CHECK(RunTest("l2v4", 409)); }
    TEST(410) { CHECK(RunTest("l2v4", 410)); }
    TEST(411) { CHECK(RunTest("l2v4", 411)); }
    TEST(412) { CHECK(RunTest("l2v4", 412)); }
    TEST(413) { CHECK(RunTest("l2v4", 413)); }
    TEST(414) { CHECK(RunTest("l2v4", 414)); }
    TEST(415) { CHECK(RunTest("l2v4", 415)); }
    TEST(416) { CHECK(RunTest("l2v4", 416)); }
    TEST(417) { CHECK(RunTest("l2v4", 417)); }
    TEST(418) { CHECK(RunTest("l2v4", 418)); }
    TEST(419) { CHECK(RunTest("l2v4", 419)); }
    TEST(420) { CHECK(RunTest("l2v4", 420)); }
    TEST(421) { CHECK(RunTest("l2v4", 421)); }
    TEST(422) { CHECK(RunTest("l2v4", 422)); }
    TEST(423) { CHECK(RunTest("l2v4", 423)); }
    TEST(424) { CHECK(RunTest("l2v4", 424)); }
    TEST(425) { CHECK(RunTest("l2v4", 425)); }
    TEST(426) { CHECK(RunTest("l2v4", 426)); }
    TEST(427) { CHECK(RunTest("l2v4", 427)); }
    TEST(428) { CHECK(RunTest("l2v4", 428)); }
    TEST(429) { CHECK(RunTest("l2v4", 429)); }
    TEST(430) { CHECK(RunTest("l2v4", 430)); }
    TEST(431) { CHECK(RunTest("l2v4", 431)); }
    TEST(432) { CHECK(RunTest("l2v4", 432)); }
    TEST(433) { CHECK(RunTest("l2v4", 433)); }
    TEST(434) { CHECK(RunTest("l2v4", 434)); }
    TEST(435) { CHECK(RunTest("l2v4", 435)); }
    TEST(436) { CHECK(RunTest("l2v4", 436)); }
    TEST(437) { CHECK(RunTest("l2v4", 437)); }
    TEST(438) { CHECK(RunTest("l2v4", 438)); }
    TEST(439) { CHECK(RunTest("l2v4", 439)); }
    TEST(440) { CHECK(RunTest("l2v4", 440)); }
    TEST(441) { CHECK(RunTest("l2v4", 441)); }
    TEST(442) { CHECK(RunTest("l2v4", 442)); }
    TEST(443) { CHECK(RunTest("l2v4", 443)); }
    TEST(444) { CHECK(RunTest("l2v4", 444)); }
    TEST(445) { CHECK(RunTest("l2v4", 445)); }
    TEST(446) { CHECK(RunTest("l2v4", 446)); }
    TEST(447) { CHECK(RunTest("l2v4", 447)); }
    TEST(448) { CHECK(RunTest("l2v4", 448)); }
    TEST(449) { CHECK(RunTest("l2v4", 449)); }
    TEST(450) { CHECK(RunTest("l2v4", 450)); }
    TEST(451) { CHECK(RunTest("l2v4", 451)); }
    TEST(452) { CHECK(RunTest("l2v4", 452)); }
    TEST(453) { CHECK(RunTest("l2v4", 453)); }
    TEST(454) { CHECK(RunTest("l2v4", 454)); }
    TEST(455) { CHECK(RunTest("l2v4", 455)); }
    TEST(456) { CHECK(RunTest("l2v4", 456)); }
    TEST(457) { CHECK(RunTest("l2v4", 457)); }
    TEST(458) { CHECK(RunTest("l2v4", 458)); }
    TEST(459) { CHECK(RunTest("l2v4", 459)); }
    TEST(460) { CHECK(RunTest("l2v4", 460)); }
    TEST(461) { CHECK(RunTest("l2v4", 461)); }
    TEST(619) { CHECK(RunTest("l2v4", 619)); }
    TEST(620) { CHECK(RunTest("l2v4", 620)); }
    TEST(621) { CHECK(RunTest("l2v4", 621)); }
    TEST(622) { CHECK(RunTest("l2v4", 622)); }
    TEST(623) { CHECK(RunTest("l2v4", 623)); }
    TEST(624) { CHECK(RunTest("l2v4", 624)); }
    TEST(634) { CHECK(RunTest("l2v4", 634)); }
    TEST(635) { CHECK(RunTest("l2v4", 635)); }
    TEST(636) { CHECK(RunTest("l2v4", 636)); }
    TEST(637) { CHECK(RunTest("l2v4", 637)); }
    TEST(638) { CHECK(RunTest("l2v4", 638)); }
    TEST(639) { CHECK(RunTest("l2v4", 639)); }
    TEST(646) { CHECK(RunTest("l2v4", 646)); }
    TEST(647) { CHECK(RunTest("l2v4", 647)); }
    TEST(648) { CHECK(RunTest("l2v4", 648)); }
    TEST(649) { CHECK(RunTest("l2v4", 649)); }
    TEST(650) { CHECK(RunTest("l2v4", 650)); }
    TEST(651) { CHECK(RunTest("l2v4", 651)); }
    TEST(652) { CHECK(RunTest("l2v4", 652)); }
    TEST(653) { CHECK(RunTest("l2v4", 653)); }
    TEST(654) { CHECK(RunTest("l2v4", 654)); }
    TEST(655) { CHECK(RunTest("l2v4", 655)); }
    TEST(656) { CHECK(RunTest("l2v4", 656)); }
    TEST(657) { CHECK(RunTest("l2v4", 657)); }
    TEST(679) { CHECK(RunTest("l2v4", 679)); }
    TEST(680) { CHECK(RunTest("l2v4", 680)); }
    TEST(681) { CHECK(RunTest("l2v4", 681)); }
    TEST(682) { CHECK(RunTest("l2v4", 682)); }
    TEST(683) { CHECK(RunTest("l2v4", 683)); }
    TEST(684) { CHECK(RunTest("l2v4", 684)); }
    TEST(689) { CHECK(RunTest("l2v4", 689)); }
    TEST(690) { CHECK(RunTest("l2v4", 690)); }
    TEST(700) { CHECK(RunTest("l2v4", 700)); }
    TEST(701) { CHECK(RunTest("l2v4", 701)); }
    TEST(702) { CHECK(RunTest("l2v4", 702)); }
    TEST(707) { CHECK(RunTest("l2v4", 707)); }
    TEST(708) { CHECK(RunTest("l2v4", 708)); }
    TEST(723) { CHECK(RunTest("l2v4", 723)); }
    TEST(724) { CHECK(RunTest("l2v4", 724)); }
    TEST(730) { CHECK(RunTest("l2v4", 730)); }
    TEST(731) { CHECK(RunTest("l2v4", 731)); }
    TEST(736) { CHECK(RunTest("l2v4", 736)); }
    TEST(737) { CHECK(RunTest("l2v4", 737)); }
    TEST(743) { CHECK(RunTest("l2v4", 743)); }
    TEST(744) { CHECK(RunTest("l2v4", 744)); }
    TEST(745) { CHECK(RunTest("l2v4", 745)); }
    TEST(746) { CHECK(RunTest("l2v4", 746)); }
    TEST(747) { CHECK(RunTest("l2v4", 747)); }
    TEST(748) { CHECK(RunTest("l2v4", 748)); }
    TEST(749) { CHECK(RunTest("l2v4", 749)); }
    TEST(750) { CHECK(RunTest("l2v4", 750)); }
    TEST(751) { CHECK(RunTest("l2v4", 751)); }
    TEST(752) { CHECK(RunTest("l2v4", 752)); }
    TEST(753) { CHECK(RunTest("l2v4", 753)); }
    TEST(754) { CHECK(RunTest("l2v4", 754)); }
    TEST(755) { CHECK(RunTest("l2v4", 755)); }
    TEST(756) { CHECK(RunTest("l2v4", 756)); }
    TEST(757) { CHECK(RunTest("l2v4", 757)); }
    TEST(758) { CHECK(RunTest("l2v4", 758)); }
    TEST(759) { CHECK(RunTest("l2v4", 759)); }
 //TEST(760) { CHECK(RunTest("l2v4", 760)); } // AlgebraicRule
 //TEST(761) { CHECK(RunTest("l2v4", 761)); } // AlgebraicRule
 //TEST(762) { CHECK(RunTest("l2v4", 762)); } // AlgebraicRule
    TEST(763) { CHECK(RunTest("l2v4", 763)); }
    TEST(764) { CHECK(RunTest("l2v4", 764)); }
    TEST(765) { CHECK(RunTest("l2v4", 765)); }
    TEST(766) { CHECK(RunTest("l2v4", 766)); }
    TEST(767) { CHECK(RunTest("l2v4", 767)); }
    TEST(768) { CHECK(RunTest("l2v4", 768)); }
    TEST(769) { CHECK(RunTest("l2v4", 769)); }
    TEST(770) { CHECK(RunTest("l2v4", 770)); }
    TEST(771) { CHECK(RunTest("l2v4", 771)); }
    TEST(772) { CHECK(RunTest("l2v4", 772)); }
    TEST(773) { CHECK(RunTest("l2v4", 773)); }
    TEST(774) { CHECK(RunTest("l2v4", 774)); }
    TEST(775) { CHECK(RunTest("l2v4", 775)); }
    TEST(776) { CHECK(RunTest("l2v4", 776)); }

    TEST(933) { CHECK(RunTest("l3v1", 933)); }
    TEST(952) { CHECK(RunTest("l3v1", 952)); }
    TEST(953) { CHECK(RunTest("l3v1", 953)); }
    TEST(954) { CHECK(RunTest("l2v4", 954)); }
    TEST(955) { CHECK(RunTest("l2v4", 955)); }
    TEST(956) { CHECK(RunTest("l2v4", 956)); }
    TEST(957) { CHECK(RunTest("l2v4", 957)); }
    TEST(958) { CHECK(RunTest("l2v4", 958)); }
    TEST(959) { CHECK(RunTest("l2v4", 959)); }
    TEST(960) { CHECK(RunTest("l3v1", 960)); }
    TEST(961) { CHECK(RunTest("l3v1", 961)); }
    TEST(962) { CHECK(RunTest("l3v1", 962)); }


    TEST(963) { CHECK(RunTest("l3v1", 963)); }
    TEST(964) { CHECK(RunTest("l3v1", 964)); }
    TEST(965) { CHECK(RunTest("l3v1", 965)); }
    TEST(966) { CHECK(RunTest("l3v1", 966)); }
    TEST(967) { CHECK(RunTest("l3v1", 967)); }
    TEST(968) { CHECK(RunTest("l2v4", 968)); }
}

SUITE(SBML_TEST_SUITE_LONGTIME)
{
    TEST(964) { CHECK(RunTest("l3v1", 964)); }
    TEST(952) { CHECK(RunTest("l3v1", 952)); }
    TEST(953) { CHECK(RunTest("l3v1", 953)); }
    TEST(954) { CHECK(RunTest("l2v4", 954)); }
    TEST(963) { CHECK(RunTest("l3v1", 963)); }
    TEST(221) { CHECK(RunTest("l2v4", 221)); }
    TEST(831) { CHECK(RunTest("l2v4", 831)); }
    TEST(326) { CHECK(RunTest("l2v4", 326)); }
    TEST(430) { CHECK(RunTest("l2v4", 430)); }
    TEST(110) { CHECK(RunTest("l2v4", 110)); }
    TEST(1116){ CHECK(RunTest("l3v1", 1116 )); }
    TEST(706) { CHECK(RunTest("l2v4", 706)); }
    TEST(579) { CHECK(RunTest("l2v4", 579)); }
    TEST(967) { CHECK(RunTest("l3v1", 967)); }
    TEST(965) { CHECK(RunTest("l3v1", 965)); }
    TEST(966) { CHECK(RunTest("l3v1", 966)); }
}


SUITE(SBML_TEST_SUITE_C_FAIL)
{
    TEST(39)  { CHECK_EQUAL(false, RunTest("l2v4", 39));  } // AlgebraicRule
    TEST(40)  { CHECK_EQUAL(false, RunTest("l2v4", 40));  } // AlgebraicRule
    TEST(182) { CHECK_EQUAL(false, RunTest("l2v4", 182)); } // AlgebraicRule
    TEST(184) { CHECK_EQUAL(false, RunTest("l2v4", 184)); } // AlgebraicRule
    TEST(531) { CHECK_EQUAL(false, RunTest("l2v4", 531)); } // AlgebraicRule
    TEST(533) { CHECK_EQUAL(false, RunTest("l2v4", 533)); } // AlgebraicRule
    TEST(534) { CHECK_EQUAL(false, RunTest("l2v4", 534)); } // AlgebraicRule
    TEST(535) { CHECK_EQUAL(false, RunTest("l2v4", 535)); } // AlgebraicRule
    TEST(536) { CHECK_EQUAL(false, RunTest("l2v4", 536)); } // AlgebraicRule
    TEST(537) { CHECK_EQUAL(false, RunTest("l2v4", 537)); } // AlgebraicRule
    TEST(538) { CHECK_EQUAL(false, RunTest("l2v4", 538)); } // AlgebraicRule
    TEST(543) { CHECK_EQUAL(false, RunTest("l2v4", 543)); } // AlgebraicRule
    TEST(546) { CHECK_EQUAL(false, RunTest("l2v4", 546)); } // AlgebraicRule
    TEST(548) { CHECK_EQUAL(false, RunTest("l2v4", 548)); } // AlgebraicRule
    TEST(549) { CHECK_EQUAL(false, RunTest("l2v4", 549)); } // AlgebraicRule
    TEST(550) { CHECK_EQUAL(false, RunTest("l2v4", 550)); } // AlgebraicRule
    TEST(551) { CHECK_EQUAL(false, RunTest("l2v4", 551)); } // AlgebraicRule
    TEST(552) { CHECK_EQUAL(false, RunTest("l2v4", 552)); } // AlgebraicRule
    TEST(553) { CHECK_EQUAL(false, RunTest("l2v4", 553)); } // AlgebraicRule
    TEST(554) { CHECK_EQUAL(false, RunTest("l2v4", 554)); } // AlgebraicRule
    TEST(555) { CHECK_EQUAL(false, RunTest("l2v4", 555)); } // AlgebraicRule
    TEST(556) { CHECK_EQUAL(false, RunTest("l2v4", 556)); } // AlgebraicRule
    TEST(557) { CHECK_EQUAL(false, RunTest("l2v4", 557)); } // AlgebraicRule
    TEST(558) { CHECK_EQUAL(false, RunTest("l2v4", 558)); } // AlgebraicRule
    TEST(559) { CHECK_EQUAL(false, RunTest("l2v4", 559)); } // AlgebraicRule
    TEST(560) { CHECK_EQUAL(false, RunTest("l2v4", 560)); } // AlgebraicRule
    TEST(561) { CHECK_EQUAL(false, RunTest("l2v4", 561)); } // AlgebraicRule
    TEST(562) { CHECK_EQUAL(false, RunTest("l2v4", 562)); } // AlgebraicRule
    TEST(563) { CHECK_EQUAL(false, RunTest("l2v4", 563)); } // AlgebraicRule
    TEST(564) { CHECK_EQUAL(false, RunTest("l2v4", 564)); } // AlgebraicRule
    TEST(565) { CHECK_EQUAL(false, RunTest("l2v4", 565)); } // AlgebraicRule
    TEST(566) { CHECK_EQUAL(false, RunTest("l2v4", 566)); } // AlgebraicRule
    TEST(567) { CHECK_EQUAL(false, RunTest("l2v4", 567)); } // AlgebraicRule
    TEST(569) { CHECK_EQUAL(false, RunTest("l2v4", 569)); } // AlgebraicRule
    TEST(570) { CHECK_EQUAL(false, RunTest("l2v4", 570)); } // AlgebraicRule
    TEST(571) { CHECK_EQUAL(false, RunTest("l2v4", 571)); } // AlgebraicRule
    TEST(573) { CHECK_EQUAL(false, RunTest("l2v4", 573)); } // AlgebraicRule
    TEST(575) { CHECK_EQUAL(false, RunTest("l2v4", 575)); } // AlgebraicRule
    TEST(576) { CHECK_EQUAL(false, RunTest("l2v4", 576)); } // AlgebraicRule

    TEST(613) { CHECK_EQUAL(false, RunTest("l2v4", 613)); } // AlgebraicRule
    TEST(614) { CHECK_EQUAL(false, RunTest("l2v4", 614)); } // AlgebraicRule
    TEST(615) { CHECK_EQUAL(false, RunTest("l2v4", 615)); } // AlgebraicRule

    TEST(628) { CHECK_EQUAL(false, RunTest("l2v4", 628)); } // AlgebraicRule
    TEST(629) { CHECK_EQUAL(false, RunTest("l2v4", 629)); } // AlgebraicRule
    TEST(630) { CHECK_EQUAL(false, RunTest("l2v4", 630)); } // AlgebraicRule

    TEST(658) { CHECK_EQUAL(false, RunTest("l2v4", 658)); } // AlgebraicRule
    TEST(659) { CHECK_EQUAL(false, RunTest("l2v4", 659)); } // AlgebraicRule
    TEST(660) { CHECK_EQUAL(false, RunTest("l2v4", 660)); } // AlgebraicRule
    TEST(661) { CHECK_EQUAL(false, RunTest("l2v4", 661)); } // AlgebraicRule
    TEST(662) { CHECK_EQUAL(false, RunTest("l2v4", 662)); } // AlgebraicRule
    TEST(663) { CHECK_EQUAL(false, RunTest("l2v4", 663)); } // AlgebraicRule
    TEST(664) { CHECK_EQUAL(false, RunTest("l2v4", 664)); } // AlgebraicRule
    TEST(665) { CHECK_EQUAL(false, RunTest("l2v4", 665)); } // AlgebraicRule
    TEST(666) { CHECK_EQUAL(false, RunTest("l2v4", 666)); } // AlgebraicRule

    TEST(673) { CHECK_EQUAL(false, RunTest("l2v4", 673)); } // AlgebraicRule
    TEST(674) { CHECK_EQUAL(false, RunTest("l2v4", 674)); } // AlgebraicRule
    TEST(675) { CHECK_EQUAL(false, RunTest("l2v4", 675)); } // AlgebraicRule

    TEST(687) { CHECK_EQUAL(false, RunTest("l2v4", 687)); } // AlgebraicRule

    TEST(695) { CHECK_EQUAL(false, RunTest("l2v4", 695)); } // AlgebraicRule
    TEST(696) { CHECK_EQUAL(false, RunTest("l2v4", 696)); } // AlgebraicRule

    TEST(705) { CHECK_EQUAL(false, RunTest("l2v4", 705)); } // AlgebraicRule

    TEST(760) { CHECK_EQUAL(false, RunTest("l2v4", 760)); } // AlgebraicRule
    TEST(761) { CHECK_EQUAL(false, RunTest("l2v4", 761)); } // AlgebraicRule
    TEST(762) { CHECK_EQUAL(false, RunTest("l2v4", 762)); } // AlgebraicRule

    TEST(777) { CHECK_EQUAL(false, RunTest("l2v4", 777)); } // AlgebraicRule
    TEST(778) { CHECK_EQUAL(false, RunTest("l2v4", 778)); } // AlgebraicRule
    TEST(779) { CHECK_EQUAL(false, RunTest("l2v4", 779)); } // AlgebraicRule
    TEST(780) { CHECK_EQUAL(false, RunTest("l2v4", 780)); } // AlgebraicRule

    TEST(844) { CHECK_EQUAL(false, RunTest("l2v4", 844)); } // AlgebraicRule

    TEST(870) { CHECK_EQUAL(false, RunTest("l2v4", 870)); } // FastReaction
    TEST(871) { CHECK_EQUAL(false, RunTest("l2v4", 871)); } // FastReaction
    TEST(872) { CHECK_EQUAL(false, RunTest("l2v4", 872)); } // FastReaction
    TEST(873) { CHECK_EQUAL(false, RunTest("l2v4", 873)); } // FastReaction
    TEST(874) { CHECK_EQUAL(false, RunTest("l2v4", 874)); } // FastReaction
    TEST(875) { CHECK_EQUAL(false, RunTest("l2v4", 875)); } // FastReaction
    TEST(876) { CHECK_EQUAL(false, RunTest("l2v4", 876)); } // AlgebraicRule

    TEST(937) { CHECK_EQUAL(false, RunTest("l2v4", 937)); } // CSymbolDelay
    TEST(938) { CHECK_EQUAL(false, RunTest("l2v4", 938)); } // CSymbolDelay
    TEST(939) { CHECK_EQUAL(false, RunTest("l2v4", 939)); } // CSymbolDelay
    TEST(940) { CHECK_EQUAL(false, RunTest("l2v4", 940)); } // CSymbolDelay
    TEST(941) { CHECK_EQUAL(false, RunTest("l2v4", 941)); } // CSymbolDelay
    TEST(942) { CHECK_EQUAL(false, RunTest("l2v4", 942)); } // CSymbolDelay
    TEST(943) { CHECK_EQUAL(false, RunTest("l2v4", 943)); } // CSymbolDelay

    TEST(981)  { CHECK_EQUAL(false, RunTest("l3v1", 981  )); } // CSymbolDelay
    TEST(982)  { CHECK_EQUAL(false, RunTest("l3v1", 982  )); } // CSymbolDelay
    TEST(983)  { CHECK_EQUAL(false, RunTest("l3v1", 983  )); } // AlgebraicRule // CSymbolDelay
    TEST(984)  { CHECK_EQUAL(false, RunTest("l3v1", 984  )); } // CSymbolDelay
    TEST(985)  { CHECK_EQUAL(false, RunTest("l3v1", 985  )); } // CSymbolDelay
    TEST(986)  { CHECK_EQUAL(false, RunTest("l3v1", 986  )); } // FastReaction
    TEST(987)  { CHECK_EQUAL(false, RunTest("l3v1", 987  )); } // FastReaction
    TEST(988)  { CHECK_EQUAL(false, RunTest("l3v1", 988  )); } // FastReaction

    TEST(993)  { CHECK_EQUAL(false, RunTest("l2v4", 993  )); } // AlgebraicRule

    TEST(1044) { CHECK_EQUAL(false, RunTest("l3v1", 1044 )); } // AlgebraicRule

    TEST(1051) { CHECK_EQUAL(false, RunTest("l3v1", 1051 )); } // FastReaction
    TEST(1052) { CHECK_EQUAL(false, RunTest("l3v1", 1052 )); } // FastReaction
    TEST(1053) { CHECK_EQUAL(false, RunTest("l3v1", 1053 )); } // FastReaction
    TEST(1054) { CHECK_EQUAL(false, RunTest("l3v1", 1054 )); } // AlgebraicRule

    TEST(1083) { CHECK_EQUAL(false, RunTest("l3v1", 1083 )); } // AlgebraicRule
    TEST(1084) { CHECK_EQUAL(false, RunTest("l3v1", 1084 )); } // AlgebraicRule
    TEST(1085) { CHECK_EQUAL(false, RunTest("l3v1", 1085 )); } // AlgebraicRule
    TEST(1086) { CHECK_EQUAL(false, RunTest("l3v1", 1086 )); } // AlgebraicRule

    TEST(1108) { CHECK_EQUAL(false, RunTest("l3v1", 1108 )); } // AlgebraicRule

    TEST(1142) { CHECK_EQUAL(false, RunTest("l3v1", 1142 )); } // AlgebraicRule // CSymbolDelay

    TEST(1173) { CHECK_EQUAL(false, RunTest("l3v1", 1173 )); } // CSymbolDelay
    TEST(1174) { CHECK_EQUAL(false, RunTest("l3v1", 1174 )); } // AlgebraicRule // CSymbolDelay
}


SUITE(SBML_TEST_SUITE_COMP)
{
TEST(1124) { CHECK(RunTest("l3v1", 1124 )); }
TEST(1125) { CHECK(RunTest("l3v1", 1125 )); }
TEST(1126) { CHECK(RunTest("l3v1", 1126 )); }
TEST(1127) { CHECK(RunTest("l3v1", 1127 )); }
TEST(1128) { CHECK(RunTest("l3v1", 1128 )); }
TEST(1129) { CHECK(RunTest("l3v1", 1129 )); }
TEST(1130) { CHECK(RunTest("l3v1", 1130 )); }
TEST(1131) { CHECK(RunTest("l3v1", 1131 )); }
TEST(1132) { CHECK(RunTest("l3v1", 1132 )); }
TEST(1133) { CHECK(RunTest("l3v1", 1133 )); }
TEST(1134) { CHECK(RunTest("l3v1", 1134 )); }
TEST(1135) { CHECK(RunTest("l3v1", 1135 )); }
TEST(1136) { CHECK(RunTest("l3v1", 1136 )); }
TEST(1137) { CHECK(RunTest("l3v1", 1137 )); }
TEST(1138) { CHECK(RunTest("l3v1", 1138 )); }
TEST(1139) { CHECK(RunTest("l3v1", 1139 )); }
TEST(1140) { CHECK(RunTest("l3v1", 1140 )); }
TEST(1141) { CHECK(RunTest("l3v1", 1141 )); }
//TEST(1142) { CHECK(RunTest("l3v1", 1142 )); } // AlgebraicRule // CSymbolDelay
TEST(1143) { CHECK(RunTest("l3v1", 1143 )); }
TEST(1144) { CHECK(RunTest("l3v1", 1144 )); }
TEST(1145) { CHECK(RunTest("l3v1", 1145 )); }
TEST(1146) { CHECK(RunTest("l3v1", 1146 )); }
TEST(1147) { CHECK(RunTest("l3v1", 1147 )); }
TEST(1148) { CHECK(RunTest("l3v1", 1148 )); }
TEST(1149) { CHECK(RunTest("l3v1", 1149 )); }
TEST(1150) { CHECK(RunTest("l3v1", 1150 )); }
TEST(1151) { CHECK(RunTest("l3v1", 1151 )); }
TEST(1152) { CHECK(RunTest("l3v1", 1152 )); }
TEST(1153) { CHECK(RunTest("l3v1", 1153 )); }
TEST(1154) { CHECK(RunTest("l3v1", 1154 )); }
TEST(1155) { CHECK(RunTest("l3v1", 1155 )); }
TEST(1156) { CHECK(RunTest("l3v1", 1156 )); }
TEST(1157) { CHECK(RunTest("l3v1", 1157 )); }
TEST(1158) { CHECK(RunTest("l3v1", 1158 )); }
TEST(1159) { CHECK(RunTest("l3v1", 1159 )); }
TEST(1160) { CHECK(RunTest("l3v1", 1160 )); }
TEST(1161) { CHECK(RunTest("l3v1", 1161 )); }
TEST(1162) { CHECK(RunTest("l3v1", 1162 )); }
TEST(1163) { CHECK(RunTest("l3v1", 1163 )); }
TEST(1164) { CHECK(RunTest("l3v1", 1164 )); }
TEST(1165) { CHECK(RunTest("l3v1", 1165 )); }
TEST(1166) { CHECK(RunTest("l3v1", 1166 )); }
TEST(1167) { CHECK(RunTest("l3v1", 1167 )); }
TEST(1168) { CHECK(RunTest("l3v1", 1168 )); }
TEST(1169) { CHECK(RunTest("l3v1", 1169 )); }
TEST(1170) { CHECK(RunTest("l3v1", 1170 )); }
TEST(1171) { CHECK(RunTest("l3v1", 1171 )); }
TEST(1172) { CHECK(RunTest("l3v1", 1172 )); }
//TEST(1173) { CHECK(RunTest("l3v1", 1173 )); } // CSymbolDelay
//TEST(1174) { CHECK(RunTest("l3v1", 1174 )); } // AlgebraicRule // CSymbolDelay
TEST(1175) { CHECK(RunTest("l3v1", 1175 )); }
TEST(1176) { CHECK(RunTest("l3v1", 1176 )); } // CSymbolDelay
TEST(1177) { CHECK(RunTest("l3v1", 1177 )); }
TEST(1178) { CHECK(RunTest("l3v1", 1178 )); }
TEST(1179) { CHECK(RunTest("l3v1", 1179 )); }
TEST(1180) { CHECK(RunTest("l3v1", 1180 )); }
TEST(1181) { CHECK(RunTest("l3v1", 1181 )); }
TEST(1182) { CHECK(RunTest("l3v1", 1182 )); }
TEST(1183) { CHECK(RunTest("l3v1", 1183 )); }
TEST(1184) { CHECK(RunTest("l3v1", 1184 )); }
TEST(1185) { CHECK(RunTest("l3v1", 1185 )); }
}

SUITE(SBML_TEST_SUITE_FBC)
{
TEST(1186) { CHECK(RunTest("l3v1", 1186 )); } // FBC
TEST(1187) { CHECK(RunTest("l3v1", 1187 )); } // FBC
TEST(1188) { CHECK(RunTest("l3v1", 1188 )); } // FBC
TEST(1189) { CHECK(RunTest("l3v1", 1189 )); } // FBC
TEST(1190) { CHECK(RunTest("l3v1", 1190 )); } // FBC
TEST(1191) { CHECK(RunTest("l3v1", 1191 )); } // FBC
TEST(1192) { CHECK(RunTest("l3v1", 1192 )); } // FBC
TEST(1193) { CHECK(RunTest("l3v1", 1193 )); } // FBC
TEST(1194) { CHECK(RunTest("l3v1", 1194 )); } // FBC
TEST(1195) { CHECK(RunTest("l3v1", 1195 )); } // FBC
TEST(1196) { CHECK(RunTest("l3v1", 1196 )); } // FBC

/*
* SBML test suite 3.3.0
*/

//FBC v2 versions of the previously-added FBC v1 tests
//TEST(1606) { CHECK(RunTest("l3v2", 1606)); }
//TEST(1607) { CHECK(RunTest("l3v2", 1607)); }
//TEST(1608) { CHECK(RunTest("l3v2", 1608)); }
//TEST(1609) { CHECK(RunTest("l3v2", 1609)); }
//TEST(1610) { CHECK(RunTest("l3v2", 1610)); }
//TEST(1611) { CHECK(RunTest("l3v2", 1611)); }
//TEST(1612) { CHECK(RunTest("l3v2", 1612)); }
//TEST(1613) { CHECK(RunTest("l3v2", 1613)); }
//TEST(1614) { CHECK(RunTest("l3v2", 1614)); }
//TEST(1615) { CHECK(RunTest("l3v2", 1615)); }
//TEST(1616) { CHECK(RunTest("l3v2", 1616)); }
//
////FBC v2 non-strict tests with assigned constant bounds and stoichiometries
//TEST(1617) { CHECK(RunTest("l3v2", 1617)); }
//TEST(1618) { CHECK(RunTest("l3v2", 1618)); }
//TEST(1619) { CHECK(RunTest("l3v2", 1619)); }
//TEST(1620) { CHECK(RunTest("l3v2", 1620)); }
//TEST(1621) { CHECK(RunTest("l3v2", 1621)); }
//TEST(1622) { CHECK(RunTest("l3v2", 1622)); }
//TEST(1623) { CHECK(RunTest("l3v2", 1623)); }
//
////FBC minimization where one reaction has a fixed value (both v1 and v2)
//TEST(1624) { CHECK(RunTest("l3v2", 1624)); }
//TEST(1625) { CHECK(RunTest("l3v2", 1625)); }
//
////FBC initial assignments and assignment rules with no MathML
//TEST(1628) { CHECK(RunTest("l3v2", 1628)); }
//TEST(1629) { CHECK(RunTest("l3v2", 1629)); }
//TEST(1630) { CHECK(RunTest("l3v2", 1630)); }

}
