#include "unit_test/UnitTest++.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrTestSuiteModelSimulation.h"
using namespace UnitTest;
using namespace rr;

extern string 			gSBMLModelsPath;
extern string 			gCompiler;
extern string 			gSupportCodeFolder;
extern string 			gTempFolder;
extern string 			gTSModelsPath;
//extern string 			gDataOutputFolder;

bool RunTest(const string& version, int number);
SUITE(SBML_l2v4)
{
TEST(1) { CHECK(RunTest("l2v4", 1)); }
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
//TEST(39) { CHECK(RunTest("l2v4", 39)); }
//TEST(40) { CHECK(RunTest("l2v4", 40)); }
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
//TEST(182) { CHECK(RunTest("l2v4", 182)); }
TEST(183) { CHECK(RunTest("l2v4", 183)); }
//TEST(184) { CHECK(RunTest("l2v4", 184)); }
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
//TEST(531) { CHECK(RunTest("l2v4", 531)); }
TEST(532) { CHECK(RunTest("l2v4", 532)); }
//TEST(533) { CHECK(RunTest("l2v4", 533)); }
//TEST(534) { CHECK(RunTest("l2v4", 534)); }
//TEST(535) { CHECK(RunTest("l2v4", 535)); }
//TEST(536) { CHECK(RunTest("l2v4", 536)); }
//TEST(537) { CHECK(RunTest("l2v4", 537)); }
//TEST(538) { CHECK(RunTest("l2v4", 538)); }
TEST(539) { CHECK(RunTest("l2v4", 539)); }
TEST(540) { CHECK(RunTest("l2v4", 540)); }
TEST(541) { CHECK(RunTest("l2v4", 541)); }
TEST(542) { CHECK(RunTest("l2v4", 542)); }
//TEST(543) { CHECK(RunTest("l2v4", 543)); }
TEST(544) { CHECK(RunTest("l2v4", 544)); }
TEST(545) { CHECK(RunTest("l2v4", 545)); }
//TEST(546) { CHECK(RunTest("l2v4", 546)); }
TEST(547) { CHECK(RunTest("l2v4", 547)); }
//TEST(548) { CHECK(RunTest("l2v4", 548)); }
//TEST(549) { CHECK(RunTest("l2v4", 549)); }
//TEST(550) { CHECK(RunTest("l2v4", 550)); }
//TEST(551) { CHECK(RunTest("l2v4", 551)); }
//TEST(552) { CHECK(RunTest("l2v4", 552)); }
//TEST(553) { CHECK(RunTest("l2v4", 553)); } //Started failing with
//TEST(554) { CHECK(RunTest("l2v4", 554)); }
//TEST(555) { CHECK(RunTest("l2v4", 555)); }
//TEST(556) { CHECK(RunTest("l2v4", 556)); }
//TEST(557) { CHECK(RunTest("l2v4", 557)); }
//TEST(558) { CHECK(RunTest("l2v4", 558)); }
//TEST(559) { CHECK(RunTest("l2v4", 559)); }
//TEST(560) { CHECK(RunTest("l2v4", 560)); }
//TEST(561) { CHECK(RunTest("l2v4", 561)); }
//TEST(562) { CHECK(RunTest("l2v4", 562)); }
//TEST(563) { CHECK(RunTest("l2v4", 563)); }
//TEST(564) { CHECK(RunTest("l2v4", 564)); }
//TEST(565) { CHECK(RunTest("l2v4", 565)); }
//TEST(566) { CHECK(RunTest("l2v4", 566)); }
//TEST(567) { CHECK(RunTest("l2v4", 567)); }
TEST(568) { CHECK(RunTest("l2v4", 568)); }
//TEST(569) { CHECK(RunTest("l2v4", 569)); }
//TEST(570) { CHECK(RunTest("l2v4", 570)); }
//TEST(571) { CHECK(RunTest("l2v4", 571)); }
TEST(572) { CHECK(RunTest("l2v4", 572)); }
//TEST(573) { CHECK(RunTest("l2v4", 573)); }
TEST(574) { CHECK(RunTest("l2v4", 574)); }
//TEST(575) { CHECK(RunTest("l2v4", 575)); }
//TEST(576) { CHECK(RunTest("l2v4", 576)); }
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
//TEST(613) { CHECK(RunTest("l2v4", 613)); }
//TEST(614) { CHECK(RunTest("l2v4", 614)); }
//TEST(615) { CHECK(RunTest("l2v4", 615)); }
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
//TEST(628) { CHECK(RunTest("l2v4", 628)); }
//TEST(629) { CHECK(RunTest("l2v4", 629)); }
//TEST(630) { CHECK(RunTest("l2v4", 630)); }
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
//TEST(658) { CHECK(RunTest("l2v4", 658)); }
//TEST(659) { CHECK(RunTest("l2v4", 659)); }
//TEST(660) { CHECK(RunTest("l2v4", 660)); }
//TEST(661) { CHECK(RunTest("l2v4", 661)); }
//TEST(662) { CHECK(RunTest("l2v4", 662)); }
//TEST(663) { CHECK(RunTest("l2v4", 663)); }
//TEST(664) { CHECK(RunTest("l2v4", 664)); }
//TEST(665) { CHECK(RunTest("l2v4", 665)); }
//TEST(666) { CHECK(RunTest("l2v4", 666)); }
TEST(667) { CHECK(RunTest("l2v4", 667)); }
TEST(668) { CHECK(RunTest("l2v4", 668)); }
TEST(669) { CHECK(RunTest("l2v4", 669)); }
TEST(670) { CHECK(RunTest("l2v4", 670)); }
TEST(671) { CHECK(RunTest("l2v4", 671)); }
TEST(672) { CHECK(RunTest("l2v4", 672)); }
//TEST(673) { CHECK(RunTest("l2v4", 673)); }
//TEST(674) { CHECK(RunTest("l2v4", 674)); }
//TEST(675) { CHECK(RunTest("l2v4", 675)); }
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
//TEST(687) { CHECK(RunTest("l2v4", 687)); }
TEST(688) { CHECK(RunTest("l2v4", 688)); }
TEST(689) { CHECK(RunTest("l2v4", 689)); }
TEST(690) { CHECK(RunTest("l2v4", 690)); }
TEST(691) { CHECK(RunTest("l2v4", 691)); }
TEST(692) { CHECK(RunTest("l2v4", 692)); }
TEST(693) { CHECK(RunTest("l2v4", 693)); }
TEST(694) { CHECK(RunTest("l2v4", 694)); }
//TEST(695) { CHECK(RunTest("l2v4", 695)); }
//TEST(696) { CHECK(RunTest("l2v4", 696)); }
TEST(697) { CHECK(RunTest("l2v4", 697)); }
TEST(698) { CHECK(RunTest("l2v4", 698)); }
TEST(699) { CHECK(RunTest("l2v4", 699)); }
TEST(700) { CHECK(RunTest("l2v4", 700)); }
TEST(701) { CHECK(RunTest("l2v4", 701)); }
TEST(702) { CHECK(RunTest("l2v4", 702)); }
TEST(703) { CHECK(RunTest("l2v4", 703)); }
TEST(704) { CHECK(RunTest("l2v4", 704)); }
//TEST(705) { CHECK(RunTest("l2v4", 705)); }
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
//TEST(760) { CHECK(RunTest("l2v4", 760)); }
//TEST(761) { CHECK(RunTest("l2v4", 761)); }
//TEST(762) { CHECK(RunTest("l2v4", 762)); }
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
//TEST(777) { CHECK(RunTest("l2v4", 777)); }
//TEST(778) { CHECK(RunTest("l2v4", 778)); }
//TEST(779) { CHECK(RunTest("l2v4", 779)); }
//TEST(780) { CHECK(RunTest("l2v4", 780)); }
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
//TEST(844) { CHECK(RunTest("l2v4", 844)); }
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
//TEST(870) { CHECK(RunTest("l2v4", 870)); }
//TEST(871) { CHECK(RunTest("l2v4", 871)); }
//TEST(872) { CHECK(RunTest("l2v4", 872)); }
//TEST(873) { CHECK(RunTest("l2v4", 873)); }
//TEST(874) { CHECK(RunTest("l2v4", 874)); }
//TEST(875) { CHECK(RunTest("l2v4", 875)); }
//TEST(876) { CHECK(RunTest("l2v4", 876)); }
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
TEST(928) { CHECK(RunTest("l2v4", 928)); }
TEST(929) { CHECK(RunTest("l2v4", 929)); }
#ifndef __linux
    TEST(930) { CHECK(RunTest("l2v4", 930)); }
    TEST(931) { CHECK(RunTest("l2v4", 931)); }
#endif
TEST(932) { CHECK(RunTest("l2v4", 932)); }
TEST(933) { CHECK(RunTest("l2v4", 933)); }
#ifndef __linux
    TEST(934) { CHECK(RunTest("l2v4", 934)); }
    TEST(935) { CHECK(RunTest("l2v4", 935)); }
#endif
TEST(936) { CHECK(RunTest("l2v4", 936)); }
//TEST(937) { CHECK(RunTest("l2v4", 937)); }
//TEST(938) { CHECK(RunTest("l2v4", 938)); }
//TEST(939) { CHECK(RunTest("l2v4", 939)); }
//TEST(940) { CHECK(RunTest("l2v4", 940)); }
//TEST(941) { CHECK(RunTest("l2v4", 941)); }
//TEST(942) { CHECK(RunTest("l2v4", 942)); }
//TEST(943) { CHECK(RunTest("l2v4", 943)); }
TEST(944) { CHECK(RunTest("l2v4", 944)); }
TEST(945) { CHECK(RunTest("l2v4", 945)); }
TEST(946) { CHECK(RunTest("l2v4", 946)); }
TEST(947) { CHECK(RunTest("l2v4", 947)); }
TEST(948) { CHECK(RunTest("l2v4", 948)); }
TEST(949) { CHECK(RunTest("l2v4", 949)); }
//TEST(950) { CHECK(RunTest("l2v4", 950)); }
//TEST(951) { CHECK(RunTest("l2v4", 951)); }
//TEST(952) { CHECK(RunTest("l2v4", 952)); }	//Started failing with poco shared lib in cvode
#ifndef __linux
    TEST(953) { CHECK(RunTest("l2v4", 953)); }
#endif
TEST(954) { CHECK(RunTest("l2v4", 954)); }
//TEST(955) { CHECK(RunTest("l2v4", 955)); }
TEST(956) { CHECK(RunTest("l2v4", 956)); }
//TEST(957) { CHECK(RunTest("l2v4", 957)); }
//TEST(958) { CHECK(RunTest("l2v4", 958)); }
//TEST(959) { CHECK(RunTest("l2v4", 959)); }
TEST(960) { CHECK(RunTest("l2v4", 960)); }
TEST(961) { CHECK(RunTest("l2v4", 961)); }
//TEST(962) { CHECK(RunTest("l2v4", 962)); }

#if !defined(__CODEGEARC__) && !defined(__linux)
    TEST(963) { CHECK(RunTest("l2v4", 963)); }
    TEST(964) { CHECK(RunTest("l2v4", 964)); }
#endif

#if !defined(_MSC_VER) && !defined(__linux)
TEST(965) { CHECK(RunTest("l2v4", 965)); }
#endif
//TEST(966) { CHECK(RunTest("l2v4", 966)); }
#ifndef __linux
    TEST(967) { CHECK(RunTest("l2v4", 967)); }
#endif
TEST(968) { CHECK(RunTest("l2v4", 968)); }
TEST(969) { CHECK(RunTest("l2v4", 969)); }
TEST(970) { CHECK(RunTest("l2v4", 970)); }
TEST(971) { CHECK(RunTest("l2v4", 971)); }
TEST(972) { CHECK(RunTest("l2v4", 972)); }
TEST(973) { CHECK(RunTest("l2v4", 973)); }
TEST(974) { CHECK(RunTest("l2v4", 974)); }
TEST(975) { CHECK(RunTest("l2v4", 975)); }
TEST(976) { CHECK(RunTest("l2v4", 976)); }
TEST(977) { CHECK(RunTest("l2v4", 977)); }
//TEST(978) { CHECK(RunTest("l2v4", 978)); }
TEST(979) { CHECK(RunTest("l2v4", 979)); }
TEST(980) { CHECK(RunTest("l2v4", 980)); }
}

bool RunTest(const string& version, int caseNumber)
{
    RoadRunner rr(gSupportCodeFolder, gCompiler, gTempFolder);

    try
    {
	    bool result(false);
 		cerr<<"Running Test:\t"<<caseNumber;
        string dummy;
        string logFileName;

        //Create log file name, e.g. 00001.log
        CreateTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, dummy);

        //Create subfolder for data output
        string dataOutputFolder = JoinPath(gTempFolder, GetTestSuiteSubFolderName(caseNumber));

        if(!CreateFolder(dataOutputFolder))
        {
            throw(rr::Exception("Failed creating output folder for data output"));
        }

        TestSuiteModelSimulation simulation(dataOutputFolder);

        rr.reset();
        simulation.UseEngine(&rr);

        //Setup filenames and paths...
        string modelFilePath(gTSModelsPath);
        string modelFileName;
        string settingsFileName;
        CreateTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName, settingsFileName);

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetCaseNumber(caseNumber);
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();

        if(!simulation.LoadSBMLFromFile())
        {
            Log(lError)<<"Failed loading SBML model";
			throw("Failed loading SBML model");
        }
        //Then read settings file if it exists..
        string settingsOveride("");
        if(!simulation.LoadSettings(settingsOveride))
        {
            Log(lError)<<"Failed loading SBML model settings";
			throw("Failed loading SBML model settings");
        }

        rr.computeAndAssignConservationLaws(false);

        //Then Simulate model
         if(!simulation.Simulate())
        {
            Log(lError)<<"Failed running simulation";
            throw("Failed running simulation");
        }

        //Write result
        if(!simulation.SaveResult())
        {
            //Failed to save data
            Log(lError)<<"Failed to save result";
            throw("Failed running simulation: Failed to save result");
        }

        if(!simulation.LoadReferenceData())
        {
            Log(lError)<<"Failed loading reference data";
            throw("Failed loading reference data");
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        simulation.SaveAllData();
        simulation.SaveModelAsXML(dataOutputFolder);

        cerr<<"\t"<< (result == true ? "PASS" : "FAIL")<<endl;
	 	return result;
     }
    catch(Exception& ex)
    {
        string error = ex.what();
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
    	return false;
    }

}
